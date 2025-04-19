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
from utils import fix_seed, evaluate, parse_hidden_dims, RolloutBuffer, NStepRolloutBuffer
from torch.utils.data import DataLoader, TensorDataset

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dims = (64,), batch_size = 64, 
                 activation_func=torch.tanh, n_steps=2048, n_epochs= 10, lr = 0.0003, 
                 gamma = 0.99, action_type = 'discrete', lambda_ = 0.95, clip_ratio = 0.2, vf_coef = 1.0, ent_coef = 0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_type = action_type
        if action_type == 'discrete':
            self.policy = DiscretePolicy(state_dim, action_dim, hidden_dims, activation_func).to(self.device)
        elif action_type == 'continuous':
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims, activation_func).to(self.device)
        self.state_value = StateValue(state_dim).to(self.device)
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gamma = gamma

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        self.value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr = lr)
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.buffer = RolloutBuffer()
    
    @torch.no_grad()
    def act(self, state, training = True):
        self.policy.train(training)
        state = torch.as_tensor(state, dtype=torch.float, device = self.device)
        if self.action_type == 'discrete':
            prob = self.policy(state)
            action = torch.multinomial(prob, 1) if training else torch.argmax(prob, dim = -1, keepdim = True)
        elif self.action_type == 'continuous':
            mu, sigma = self.policy(state)
            action = torch.normal(mu, sigma) if training else mu
            action = torch.tanh(action)
        return action.cpu().numpy()

    def learn(self):
        """얻은 데이터로부터 n_epochs번 update수행"""
        self.policy.train()
        self.state_value.train()
        state, action, reward, n_state, done = self.buffer.sample()
        state, action, reward, n_state, done = map(lambda x: x.to(self.device), [state, action, reward, n_state, done])

        with torch.no_grad():
            delta =  reward + (1- done) * self.gamma * self.state_value(n_state) - self.state_value(state)
            adv = torch.clone(delta)
            ret = torch.clone(reward)
            for t in reversed(range(len(reward) - 1)):
                adv[t] += (1-done[t]) * self.gamma * self.lambda_ * adv[t+1] #* GAE 식 정리하면 이렇게 나옴. 무한까지 가야하지만, batch size만큼만 감.
                ret[t] += (1-done[t]) * self.gamma * ret[t+1]

            # importance sampling을 위한 log_prob_old 계산
            if self.action_type == 'discrete':
                probs = self.policy(state)
                log_probs_old = torch.log(probs.gather(dim = 1, index = action.long()))
            elif self.action_type == 'continuous':
                mu, sigma = self.policy(state)
                normal_dist = Normal(mu, sigma)
                action = torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 - 1e-7))
                log_probs_old = normal_dist.log_prob(action).sum(dim = -1, keepdim = True) # (full batch, 1)
        
        tensor_dts = TensorDataset(state, action, ret, adv, log_probs_old)
        loader = DataLoader(tensor_dts, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.n_epochs):
            for batch in loader:
                state, action, ret, adv, log_probs_old = map(lambda x: x.to(self.device), batch) #각각이 (batch_size , eachdim)
                
                value = self.state_value(state)
                value_loss = F.mse_loss(value, ret) #* 끝까지 가면서 올바른 return값을 받을 수 있으니까.

                mu, sigma = self.policy(state)
                normal_dist = Normal(mu, sigma)
                action = torch.atanh(torch.clamp(action, -1.0 + 1e-5, 1.0 - 1e-5))
                log_probs = normal_dist.log_prob(action).sum(dim = -1, keepdim = True) #batch 별로 되어있는 차원 그대로 유지하기 위함인듯

                ratio = torch.exp(log_probs - log_probs_old) #* importance ratio
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv

                policy_loss = -torch.min(surr1, surr2).mean() #* surrogate loss
                entropy = normal_dist.entropy().mean() #* entropy loss

                loss = policy_loss - self.ent_coef * entropy + self.vf_coef * value_loss
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.state_value.parameters(), max_norm=0.5)
                self.policy_optimizer.step()
                self.value_optimizer.step()

        result = {'tot_loss' : loss.item(), 'policy_loss' : policy_loss.item(), 'value_loss' : value_loss.item(), 'entropy' : entropy.item()}
        return result
    
    def process(self, transition):
        result = None
        self.buffer.store(transition)
        if self.buffer.size >= self.n_steps: #* n_steps 수 만큼 data 모은 후 여러번 update수행
            result = self.learn()
            
        return result

def main(args):
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]  # env.observation_space.shape state의 tuple반환
    if args.action_type == 'discrete':
        action_dim = env.action_space.n  # discrete의 경우, int값, continous의 경우 tuple반환.
    else:
        action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim, hidden_dims = args.hidden_dims, batch_size = args.batch_size,
                 lr = args.lr, gamma = args.gamma, action_type=args.action_type, n_steps= args.n_steps,
                 n_epochs = args.n_epochs, lambda_ = args.lambda_, clip_ratio = args.clip_ratio, 
                 vf_coef = args.vf_coef, ent_coef = args.ent_coef)
    if args.test:
        # print(f"Loading model from {args.load_path}")
        agent.policy.load_state_dict(torch.load(f"./saved/best_{args.env_name}_{args.model}.pth", map_location=agent.device)) # agent act만 구해주면 되니까 poolicy만 load해주면 됨.
        agent.policy.eval()

        score = evaluate(args.env_name, agent, args.seed, args.eval_iteration, args.action_type, args.visualize)
        print(f"[TEST MODE] Evaluation return: {score}")
        return  # test만 하고 종료


    state, _ = env.reset(seed = args.seed)  # 각 evaluation마다 시작 state다를 수 있음.
    terminated, truncated = False, False
    best_score = -np.inf
    for t in tqdm(range(1, args.max_iteration + 1), leave = False):
        action = agent.act(state)
        if args.action_type == 'discrete':
            n_state, reward, terminated, truncated, _ = env.step(action[0])
        else:
            n_state, reward, terminated, truncated, _ = env.step(action * 2) #pendulum의 경우, action space가 -2 to 2이므로 2배를 해줌. tanh로 -1 to 1이 되었으니까.
        result = agent.process((state, action, reward, n_state, terminated, truncated))
        state = n_state

        if result is not None: #이제는 batch대로 update하므로 result가 비지 않았을때 update.
            # state, _ = env.reset() #이제는 batch마다 학습하는 것이니까 reset해주면 안됨.
            if args.wandb:
                wandb.log({
                    "batch_loss" : result['tot_loss'],
                    "batch_entropy" : result['entropy'],
                    "batch_policy_loss" : result['policy_loss'],
                    "batch_value_loss" : result['value_loss']
                })

        if terminated or truncated: #이제는 batch 대로 update하므로 episodes가 끝나면 update해줘야함.
            state, _ = env.reset()

        if t % args.eval_interval == 0:
            score = evaluate(args.env_name, agent, args.seed, args.eval_iteration, args.action_type, args.visualize)
            print(f"time : {t},eval Avg return : {score}")
            if score > best_score:
                best_score = score
                torch.save(agent.policy.state_dict(), f"./saved/best_{args.env_name}_{args.model}.pth")
            if args.wandb:
                wandb.log({
                    "eval_score" : score
                })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type = str, default = 'Pendulum-v1')
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--max_iteration', type = int, default = 1000000)
    parser.add_argument('--eval_interval', type = int, default = 10000)
    parser.add_argument('--eval_iteration', type = int, default = 10)
    parser.add_argument('--gamma', type = float, default=0.99)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--wandb', type = bool, default = False)
    parser.add_argument('--action_type', type = str, default = 'continuous')
    parser.add_argument('--hidden_dims', type = parse_hidden_dims, default = "(64, 64)") #"(32,32)" 형태로 넣어줘야함.
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--visualize', type = bool, default= False)
    parser.add_argument('--lambda_', type = float, default= 0.95)
    parser.add_argument('--actfunc', type = str, default= F.relu)
    parser.add_argument('--n_steps', type = int, default = 2048) # 여러번 update에 사용할 data num
    parser.add_argument('--n_epochs', type = int, default = 10) # 같은 데이터로 몇번 update수행할지
    parser.add_argument('--model', type = str, default = 'PPO')
    parser.add_argument('--clip_ratio', type = float, default = 0.2)
    parser.add_argument('--vf_coef', type = float, default = 1.0)
    parser.add_argument('--ent_coef', type = float, default = 0.01)
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    if args.wandb:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb.init(project = 'reinforce', group = f"{args.env_name}_PPO", 
                   name = f"{exp_name}", config = vars(args))

    fix_seed(args.seed)
    main(args)

# export CUDA_VISIBLE_DEVICES=1
# python ppo.py --wandb True
# python ppo.py --test --visualize True
