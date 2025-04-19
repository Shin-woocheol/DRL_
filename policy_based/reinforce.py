import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import argparse
import wandb
from datetime import datetime

from torch.distributions import Normal
from tqdm import tqdm
from network import DiscretePolicy, GaussianPolicy, StateValue
from utils import fix_seed, evaluate, parse_hidden_dims

class Baseline_Reinforce:
    def __init__(self, state_dim, action_dim, gamma = 0.9, lr = 0.001, action_type = 'discrete', hidden_dims = (64,)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if action_type == 'discrete':
            self.policy = DiscretePolicy(state_dim, action_dim, hidden_dims= hidden_dims).to(self.device)
        elif action_type == 'continuous':
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims= hidden_dims).to(self.device)
        self.state_value = StateValue(state_dim).to(self.device)
        self.action_type = action_type
        self.gamma = gamma
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        self.value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr = lr)
        self.buffer = [] # N step interaction info로 update.

    @torch.no_grad()
    def act(self, state, training = True):
        self.policy.train(training) # model train mode set.
        state = torch.as_tensor(state, dtype = torch.float, device = self.device) # (batch, state_dim)
        if self.action_type == 'discrete':
            prob = self.policy(state) # (batch, action_dim)
            action = torch.multinomial(prob,1) if training else torch.argmax(prob, dim = -1, keepdim=True)
            # torch.multinomial(input, num_samples, replacement=False) : input에서 num_samples개를 뽑음. replacement는 중복여부. row 기준 확률.
        else:
            mu, sigma = self.policy(state) # (batch, action_dim) # 여기서의 actiondim은 정해줘야하는 action의 dim
            action = torch.normal(mu, sigma) if training else mu #* discrete에서 eval시에는 max prob으로 뽑은 것과 같이, continuous에서도 eval에서는 mu.
            action = torch.tanh(action) #* env마다 max action, min action이 다르다. 그러므로, action값을 -1 to 1로 만들어준 후, normalization푸는 것.
        return action.cpu().numpy() # (batch, action_dim) 

    def learn(self):
        self.policy.train()
        self.state_value.train() #이거 왜 안해주지?
        state, action, reward, _, _, _ = map(np.stack, zip(*self.buffer)) # (batch, each_dim)
        state, action, reward = map(lambda x: torch.as_tensor(x, dtype=torch.float, device = self.device), [state, action, reward])
        
        if reward.ndim == 1: # reward가 1차원일 경우, (batch_size, ) -> (1, batch_size)로 바꿔줌.
            reward = reward.unsqueeze(1)

        ret = torch.clone(reward)
        for t in reversed(range(len(ret) - 1)): 
            ret[t] += self.gamma * ret[t+1]

        if self.action_type == 'discrete':
            probs = self.policy(state) 
            log_probs = torch.log(probs.gather(dim = 1, index = action.long()))
        else:
            mu, sigma = self.policy(state)
            normal_dist = Normal(mu, sigma)
            action = torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = normal_dist.log_prob(action).sum(dim = -1, keepdim = True) 

        # value predict
        state_value = self.state_value(state) # (batch_size, 1)

        policy_loss = -((ret - state_value.detach()) * log_probs).mean() #* 해당 loss를 계산할 때는 state_value가 constant와 같이 쓰여야 하므로 detach.
        #* 이렇게 해주면, 해당 value를 이용하는 것은 맞지만, 해당 value가 도출된 model의 gradient는 구하지 않음. 즉, state value model과 관련이 없게 만들어줌.
        # target이 움직이면 안됨. 
        self.policy_optimizer.zero_grad() # 기존 gradient 초기화
        policy_loss.backward() # policy loss에 대해 모델의 각 param에 대한 gradient 계산.
        self.policy_optimizer.step() # 계산된 gradient를 이용해서 모델의 param update.

        value_loss = F.mse_loss(state_value, ret) #* 해당 time에 state value function이 가져야 할 값은 return값.
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        result = {'policy_loss' : policy_loss.item(), 'value_loss' : value_loss.item()} # tensor.mean()을 해도 tensor여서 value만 받기 위해.
        return result

    def process(self, transition): #buffer에 appen를 하다가, episode종료시에는 학습 시작.
        result = None
        self.buffer.append(transition)
        if transition[-1] or transition[-2]: #terminated or truncated
            result = self.learn() #episode종료시 바로 learning 시작.
            self.buffer = [] #* episode 종료시 buffer 초기화 -> log policy에 대한 theta가 현재 theta이기 때문.
        return result

class Reinforce:
    def __init__(self, state_dim, action_dim, gamma = 0.99, lr = 0.001, action_type = 'discrete', hidden_dims = (64,)):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if action_type == 'discrete':
            self.policy = DiscretePolicy(state_dim, action_dim).to(self.device)
        elif action_type == 'continuous':
            self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.action_type = action_type
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        self.buffer = [] # N step interaction info로 update.

    @torch.no_grad()
    def act(self, state, training = True):
        self.policy.train(training) # model train mode set.
        state = torch.as_tensor(state, dtype = torch.float, device = self.device)
        if self.action_type == 'discrete':
            prob = self.policy(state)
            action = torch.multinomial(prob,1) if training else torch.argmax(prob, dim = -1, keepdim=True)
        else:
            mu, sigma = self.policy(state)
            action = torch.normal(mu, sigma) if training else mu #* discrete에서 eval시에는 max prob으로 뽑은 것과 같이, continuous에서도 eval에서는 mu.
            action = torch.tanh(action) #* env마다 max action, min action이 다르다. 그러므로, action값을 -1 to 1로 만들어준 후, normalization푸는 것.

        return action.cpu().numpy()

    def learn(self):
        '''
        reinforce의 경우, policy gradient theorem으로 구한 objective function의 미분 term을 MC estimation을
        통해서 구한 후, gradient ascent를 해주는 방식임.
        objective function의 미분값은 reinforce에선, return과 log policy의 미분값의 곱.
        '''
        self.policy.train()
        state, action, reward, _, _, _ = map(np.stack, zip(*self.buffer)) # 각각이 (batch_size, each_dim) 이런식으로 나옴.
        #buffer에는 state, action, reward, n_state, terminated, truncated 이렇게 6가지가 batch만큼 있음.
        #근데 이걸 unpacking을 하고, 즉, unpacking을 하면, 각 5가지가 한세트인 list가 argument로 들어감.
        # 근데 이거에 zip을 하게 되면, 같은 index끼리 묶여서 state끼리, action끼리, ... 묶인 튜플 5개 생김.
        # map은 첫번째 argument인 function을 element마다 반복하는 것이니까,
        # state가 연속적인 tuple, action이 연속적인 tuple마다 stack이 들어가서
        # 각 list에 state가 들어가있고 action이 들어가있고 한 것이 나옴.
        state, action, reward = map(lambda x: torch.as_tensor(x, dtype=torch.float, device = self.device), [state, action, reward])
        #각 state action reward를 float tensor로 변경.
        if reward.ndim == 1: # reward가 1차원일 경우, (batch_size, ) -> (1, batch_size)로 바꿔줌.
            reward = reward.unsqueeze(1) # (batch, 1)로 확장하는데, 나중에 2차원tensor가 사용되나봄. 아직 왜하는지 모름.
        # 나중에, ret * log_probs로 tensor 곱연산을 진행하는데, 이때 log_probs 가 batch x 1이어서
        # 이 size맞춰주려고 미리 unsqueeze를 해놓는거네.

        # 그리고 아래 코드에서도 사실 return과 reward를 반복문 후에 각각 unsqueeze해줘도 같음. 즉, 진짜 차원 맞춰주기 용.
        ret = torch.clone(reward)
        for t in reversed(range(len(ret) - 1)): # 1짜리 차원 추가한 것은 없는 것으로 봐도 무방.
            ret[t] += self.gamma * ret[t+1] # decay를 적용한 return으로 만들어줌 뒤부터 접근하면서

        if self.action_type == 'discrete':
            probs = self.policy(state) # (batch_size, action_dim)
            # discrete이므로, possible action에 대한 확률tensor.
            # state가 (batch_size x state_dim)이 들어감. 그럼 probs는 (batch_size x action_dim)
            # 그래서 아래 코드는, 해당 시점에 어떤 action을 골랏는지 buffer에서 꺼낸 것이니까,
            # 그 action을 어떤 prob으로 꺼낸 것인지를 고름.
            # dim = 1을 해서 action_dim에서 gather하도록 만듬. 그리고 그 확률을 log prob으로.
            # action은 (batch_size x action_dim) action dim쪽을 index로 사용해서 gathering
            log_probs = torch.log(probs.gather(dim = 1, index = action.long()))
            # return x log prob의 mean으로 estimate하면 그게 objective function의 derivation.
        else:
            mu, sigma = self.policy(state)
            normal_dist = Normal(mu, sigma)
            action = torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 - 1e-7)) #* 위에서 action을 받고난 후, tanh로 -1 to 1로 만들어준 것 다시 풀어줌. 그래야 원래 layer에서 나온 action이니까. clamp해주는 것은 -1 or 1에서 무한이 나오기 때문.
            log_probs = normal_dist.log_prob(action).sum(dim = -1, keepdim = True) # 해당 action이 나올 log prob를 normal dist로부터 받음. 결국 이 부분이 log policy

        policy_loss = -(ret * log_probs).mean() #gradient descent를 쓸 것이니까, 음수 취해줌. #* 결국 이 부분을 policy loss 즉, J(theta)로 본다는 것임. gradient 들어가면 그걸로 update하면 gradient descent 식과 같아지니까.
        self.optimizer.zero_grad() # 기존 gradient 초기화
        policy_loss.backward() # policy loss에 대해 모델의 각 param에 대한 gradient 계산. logpi에 대한 미분.
        self.optimizer.step() # 계산된 gradient를 이용해서 모델의 param update.

        result = {'policy_loss' : policy_loss.item()} # tensor.mean()을 해도 tensor여서 value만 받기 위해.

        return result

    def process(self, transition): #buffer에 appen를 하다가, episode종료시에는 학습 시작.
        result = None
        self.buffer.append(transition)
        if transition[-1] or transition[-2]: #terminated or truncated
            result = self.learn() #episode종료시 바로 learning 시작.
            self.buffer = [] #* episode 종료시 buffer 초기화 -> log policy에 대한 theta가 현재 theta이기 때문.
        return result


def main(args):
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]  # env.observation_space.shape state의 tuple반환
    if args.action_type == 'discrete':
        action_dim = env.action_space.n  # discrete의 경우, int값, continous의 경우 tuple반환.
    else:
        action_dim = env.action_space.shape[0]
    
    if args.model == "reinforce":
        agent = Reinforce(state_dim, action_dim, args.gamma, args.lr, args.action_type, args.hidden_dims)
    else:
        agent = Baseline_Reinforce(state_dim, action_dim, args.gamma, args.lr, args.action_type, args.hidden_dims)

    state, _ = env.reset(seed = args.seed)  # 각 evaluation마다 시작 state다를 수 있음.
    terminated, truncated = False, False
    episode = 1
    ret = 0
    best_score = -np.inf
    for t in tqdm(range(1, args.max_iteration + 1), leave = False):
        action = agent.act(state)
        if args.action_type == 'discrete':
            n_state, reward, terminated, truncated, _ = env.step(action[0])
        else:
            n_state, reward, terminated, truncated, _ = env.step(action * 2) #pendulum의 경우, action space가 -2 to 2이므로 2배를 해줌. tanh로 -1 to 1이 되었으니까.
        result = agent.process((state, action, reward, n_state, terminated, truncated))
        state = n_state
        ret += reward

        if terminated or truncated: #episode 종료 시.
            state, _ = env.reset()
            # terminated, truncated = False, False
            # epi_return = evaluate(args.env_name, agent, args.seed, args.eval_iteration, args.action_type)
            # print(f"episode : {episode}, return : {epi_return}")
            if args.wandb:
                if args.model == "reinforce":
                    wandb.log({
                        "episode_score" : ret,
                        "policy_loss" : result['policy_loss']
                    })
                elif args.model == "basereinforce":
                    wandb.log({
                        "episode_score" : ret,
                        "policy_loss" : result['policy_loss'],
                        "value_loss" : result['value_loss']
                    })
            episode += 1
            if ret > best_score:
                torch.save(agent.policy.state_dict(), f"./saved/best_{args.env_name}_{args.model}.pth")
            ret = 0
        if t % args.eval_interval == 0:
            score = evaluate(args.env_name, agent, args.seed, args.eval_iteration, args.action_type)
            print(f"time : {t},eval Avg return : {score}")
            if args.wandb:
                wandb.log({
                    "eval_score" : score
                })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = 'CartPole-v1')
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--max_iteration', type = int, default = 1000000)
    parser.add_argument('--eval_interval', type = int, default = 10000)
    parser.add_argument('--eval_iteration', type = int, default = 5)
    parser.add_argument('--gamma', type = float, default=0.99)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--wandb', type = bool, default = False)
    parser.add_argument('--action_type', type = str, default = 'discrete')
    parser.add_argument('--model', type = str, default="reinforce")
    parser.add_argument('--hidden_dims', type = parse_hidden_dims, default = (32, ))
    args = parser.parse_args()

    if args.wandb:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(project = 'reinforce', group = f"{args.env_name}_{args.model}", 
                   name=f"{exp_name}",
                   config = vars(args))

    fix_seed(args.seed)
    main(args)

# export CUDA_VISIBLE_DEVICES=1
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.9 --lr 0.0003
# gamma, lr에 따른 변동이 심함.
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.9 --lr 0.0001
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.85 --lr 0.0003
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.85 --lr 0.0001 --max_iteration 200000

# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.87 --lr 0.0005 --max_iteration 200000 #value, policy loss 0에 수렴
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.88 --lr 0.0007 --max_iteration 200000 # lr이 높아지면 1428.4074에 수렴함.
# python reinforce.py --env_name Pendulum-v1 --action_type continuous --wandb True --model basereinforce --gamma 0.83 --lr 0.0005 --max_iteration 200000 #value, policy loss 0에 수렴

# python reinforce.py --model basereinforce --wandb True