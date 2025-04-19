import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import wandb
import torch.nn.functional as F
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps_done = 0

class QNet(nn.Module):  # state를 input으로 받아서 각 action에 대한 action value를 output으로 내보냄.
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNagent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=args.memory_size)  # replay buffer
        self.discount_factor = args.discount_factor  # 다음 state에서의 action value 가치 조정. gamma
        self.epsilon = args.epsilon  # exploration
        self.epsilon_decay = args.epsilon_decay  # exploration decay해줘야 optimal policy로 수렴함.
        self.epsilon_min = args.epsilon_min
        self.target_model = QNet(state_size, action_size, args.hidden_size).to(device) # TD target 계산에 쓰이는 policy
        self.policy_model = QNet(state_size, action_size, args.hidden_size).to(device) #behavior policy
        self.target_model.load_state_dict(self.policy_model.state_dict()) #policy model 학습 가중치 target model로 복사.
        self.tau = args.tau
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=args.lr, amsgrad=True)  # 학습할 parameter
        self.tot_episode = args.episodes

    def remember(self, state, action, reward, n_state, done):
        self.memory.append((state, action, reward, n_state, done))  # deque추가 # tuple형태로 추가.

    def choose_action(self, state):  # state를 input으로 받아서 다음 action을 return, epsilon greedy.
        global steps_done
        if np.random.rand() <= self.epsilon:  # epsilon의 확률로 random action
            return random.randrange(self.action_size)  # 범위 내 정수 random
        state = torch.FloatTensor(state).to(device)  # gym에서 state numpy로 반환됨. pytorch model 입력이 tensor여야 해서 변환.
        with torch.no_grad():
            action_values = self.policy_model(state) #efficient
        steps_done += 1
        return torch.argmax(action_values).item()  # action value중 가장 큰 값의 index 반환

    def replay(self, batch_size, episode):
        global steps_done
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size) # to reduce correlation. # (batch_size,)

        # 아래와 같이 해도 됨. 주의해야할 사항은 model을 돌릴 때, batch로 넣어줘야 한다는 것.
        # states = torch.FloatTensor(np.array([x[0] for x in mini_batch])).to(device)
        # actions = torch.LongTensor(np.array([x[1] for x in mini_batch])).to(device)
        # rewards = torch.FloatTensor(np.array([x[2] for x in mini_batch])).to(device)
        # next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch])).to(device)
        # dones = torch.FloatTensor(np.array([x[4] for x in mini_batch])).to(device)

        batch = np.array(mini_batch, dtype=object) 
        states = torch.tensor(np.vstack(batch[:, 0]), dtype=torch.float32, device=device) # (batch_size, state_size)
        actions = torch.tensor(np.array(batch[:, 1].tolist()), dtype=torch.long, device=device)
        rewards = torch.tensor(np.array(batch[:, 2].tolist()), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.vstack(batch[:, 3]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(batch[:, 4].tolist()), dtype=torch.float32, device=device)

        current_qs = self.policy_model(states)  # vector로 넣으면 각 sample 마다 계산해서 나옴 batch_size x action_size 반환.
        current_qs = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)  # tensor 차원 맞춰준 후, action뽑아내고 다시 list형태.
        with torch.no_grad(): # 왜 사용하는지가 중요. 만약 아래의 것을 policy model로 사용하고 있었으면, no_grad를 사용하지 않았을 경우에 tartget또한 theta에 depend됨.
            #그래서 나중에 grad를 해주면, target과 current 둘 다 theta에 depend 되어있어서 학습이 잘 되지 않음.
            #그러므로 tartget은 current에 대해서 constant로 작동하게 하기 위해서 no_grad를 이용하여 끊어줌.
            next_qs = self.target_model(next_states).max(1)[0]  # max(1)을 하면 max값, index반환. #q learning의 target은 reward + gaama * max_a' Q(s', a')임.
        targets = rewards + (1 - dones) * self.discount_factor * next_qs  # done이면 reward만.

        criterion = nn.SmoothL1Loss()
        loss = criterion(current_qs, targets)
        # loss = nn.MSELoss()(current_qs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 100)  # gradient clipping으로 gradient exploding 방지.
        self.optimizer.step()

        #우리가 목표로 하는 policy는 target model. 그런데 한번에 크게 학습되면 변동이 클 수 있으니 policy로 weight를 바꾸고 target을 조금씩 update
        # target policy에서 구해지는 target값으로 일정 이상은 변해줘야함. 1step마다 target을 update해주면 그게 moving target이 됨. 
        target_state_dict = self.target_model.state_dict()
        policy_state_dict = self.policy_model.state_dict()
        for key in target_state_dict: #exponential moving average
            target_state_dict[key] = args.tau * policy_state_dict[key] + (1 - args.tau) * target_state_dict[key] #tau 0.005
        self.target_model.load_state_dict(target_state_dict)

        if self.epsilon > self.epsilon_min:
            self.epsilon = args.epsilon_min + (args.epsilon - args.epsilon_min) * np.exp(-1. * steps_done / args.epsilon_decay)

def main(args):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNagent(state_size, action_size,args)
    batch_size = args.batch_size

    for e in range(args.episodes):
        state = env.reset()[0]
        done = False
        time = 0
        episode_reward = 0

        while not done:
            action = agent.choose_action(state)
            n_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 종료 확인

            agent.remember(state, action, reward, n_state, done)
            state = n_state
            time += 1
            episode_reward += reward

            if done:
                print(
                    f"Episode: {e}/{args.episodes}, Score: {time}, Episod_reward : {episode_reward}, Epsilon: {agent.epsilon:.3f}")
                wandb.log({
                    "Episode": e,
                    "Score": time,
                    "Episode_reward": episode_reward,
                    "Epsilon": agent.epsilon
                })
                break

            if len(agent.memory) > batch_size and e % args.update_period == 0:
                agent.replay(batch_size, e)  # 에이전트 학습
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type =int, default = 1000)
    parser.add_argument('--lr', type =float, default = 0.0001)
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--update_period', type = int, default = 1)
    parser.add_argument('--discount_factor', type = float, default = 0.99)
    parser.add_argument('--epsilon', type = float, default = 0.9)
    parser.add_argument('--epsilon_decay', type = int, default = 1000)
    parser.add_argument('--memory_size', type = int, default = 10000)
    parser.add_argument('--epsilon_min', type = float, default = 0.05)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--comment', type = str, default = None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--group', type=str, default = None)
    parser.add_argument('--tau', type=float, default=0.005)
    args = parser.parse_args()

    wandb.init(project="cartpole", name=args.name, group=args.group, config=vars(args))

    main(args)

#python cartpole.py --episodes 1000 --lr 0.0001 --batch_size 128 --update_period 1 --discount_factor 0.99 --epsilon 0.9 --epsilon_decay 1000 --memory_size 10000 --epsilon_min 0.05 --hidden_size 128 --group cartpole