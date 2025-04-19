import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import argparse
import wandb

from torch.distributions import Normal
from tqdm import tqdm
from network import DiscretePolicy, GaussianPolicy, StateValue
from utils import fix_seed, evaluate, parse_hidden_dims, RolloutBuffer, NStepRolloutBuffer

class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dims = (64,), batch_size = 32, activation_func=F.relu, lr = 0.0003, gamma = 0.99, action_type = 'discrete', lambda_ = 0.95):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_type = action_type
        if action_type == 'discrete':
            self.policy = DiscretePolicy(state_dim, action_dim, hidden_dims, activation_func).to(self.device)
        elif action_type == 'continuous':
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims, activation_func).to(self.device)
        self.state_value = StateValue(state_dim).to(self.device)
        self.gamma = gamma
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr)
        self.value_optimizer = torch.optim.Adam(self.state_value.parameters(), lr = lr)
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.buffer = RolloutBuffer() #* GAE는 무한 step까지의 n-step TD error를 exponential weighted sum해주는 것이니까.

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
        self.policy.train()
        self.state_value.train()
        state, action, reward, n_state, done = self.buffer.sample()
        state, action, reward, n_state, done = map(lambda x: x.to(self.device), [state, action, reward, n_state, done])

        # GAE
        with torch.no_grad():
            delta =  reward + (1- done) * self.gamma * self.state_value(n_state) - self.state_value(state)
            adv = torch.clone(delta)
            ret = torch.clone(reward)
            for t in reversed(range(len(reward) - 1)):
                adv[t] += (1-done[t]) * self.gamma * self.lambda_ * adv[t+1] #* GAE 식 정리하면 이렇게 나옴. 무한까지 가야하지만, batch size만큼만 감.
                ret[t] += (1-done[t]) * self.gamma * ret[t+1]

        if self.action_type == 'discrete':
            probs = self.policy(state)
            log_probs = torch.log(probs.gather(dim = 1, index = action.long()))
        elif self.action_type == 'continuous':
            mu, sigma = self.policy(state)
            normal_dist = Normal(mu, sigma)
            action = torch.atanh(torch.clamp(action, -1.0 + 1e-7, 1.0 - 1e-7))
            log_probs = normal_dist.log_prob(action).sum(dim = -1, keepdim = True) #batch 별로 되어있는 차원 그대로 유지하기 위함인듯
        
        #value network update
        value = self.state_value(state)
        value_loss = F.mse_loss(value, ret) #* 끝까지 가면서 올바른 return값을 받을 수 있으니까.
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        #policy network update
        #* 기존 Q function의 정의에 따라 r + gamma * value function을 Q function 대신 이용, value는 baseline.
        policy_loss = -(log_probs * adv).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        result = {'policy_loss' : policy_loss.item(), 'value_loss' : value_loss.item()}
        return result
    
    def process(self, transition):
        result = None
        self.buffer.store(transition)
        if self.buffer.size >= self.batch_size: #* batch_size를 1로 하면 online actor-critic, N으로 하면 batch actor-critic
            result = self.learn()
            
        return result


def main(args):
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]  # env.observation_space.shape state의 tuple반환
    if args.action_type == 'discrete':
        action_dim = env.action_space.n  # discrete의 경우, int값, continous의 경우 tuple반환.
    else:
        action_dim = env.action_space.shape[0]

    agent = ActorCritic(state_dim, action_dim, args.hidden_dims, args.batch_size, lr = args.lr, gamma = args.gamma, action_type=args.action_type, lambda_= args.lambda_, activation_func=args.actfunc)

    state, _ = env.reset(seed = args.seed)  # 각 evaluation마다 시작 state다를 수 있음.
    terminated, truncated = False, False
    
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
                    "batch_policy_loss" : result['policy_loss'],
                    "batch_value_loss" : result['value_loss']
                })

        if terminated or truncated: #이제는 batch 대로 update하므로 episodes가 끝나면 update해줘야함.
            state, _ = env.reset()

        if t % args.eval_interval == 0:
            score = evaluate(args.env_name, agent, args.seed, args.eval_iteration, args.action_type, args.visualize)
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
    parser.add_argument('--eval_iteration', type = int, default = 10)
    parser.add_argument('--gamma', type = float, default=0.99)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--wandb', type = bool, default = False)
    parser.add_argument('--action_type', type = str, default = 'discrete')
    parser.add_argument('--hidden_dims', type = parse_hidden_dims, default = (32,)) #"(32,32)" 형태로 넣어줘야함.
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--visualize', type = bool, default= False)
    parser.add_argument('--lambda_', type = float, default= 0.95)
    parser.add_argument('--actfunc', type = str, default= F.relu)
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project = 'reinforce', group = f"{args.env_name}_GAEActorCritic", config = vars(args))

    fix_seed(args.seed)
    main(args)

# python actor-critic.py --env_name Pendulum-v1 --action_type continuous
# python actor-critic.py --env_name Pendulum-v1 --action_type continuous --batch_size 1 # online learning
# python actor-critic.py --env_name Pendulum-v1 --action_type continuous --hidden_dims "(64,64)" --wandb True
# python actor-critic.py --env_name Pendulum-v1 --action_type continuous --hidden_dims "(64,64)" --wandb True --batch_size 64 --gamma 0.95 --max_iteration 200000
# python actor-critic.py --env_name Pendulum-v1 --action_type continuous --hidden_dims "(64,64)" --wandb True --batch_size 1 --gamma 0.95 --max_iteration 200000
# python actor-critic.py --wandb True --max_iteration 200000


# python gae_actor-critic.py --env_name Pendulum-v1 --action_type continuous --hidden_dims "(64,64)" --wandb True --batch_size 64 --gamma 0.90 --max_iteration 1000000 --eval_interval 10000 --lambda_ 0.7 --actfunc F.tanh