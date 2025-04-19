import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import wandb
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps_done = 0

class QNet(nn.Module):  # state를 input으로 받아서 각 action에 대한 action value를 output으로 내보냄.
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),  # 64정도로 늘리기
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.model(state)

class DQNagent:
    def __init__(self, state_size, action_size, args):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=args.memory_size)  # replay buffer
        self.discount_factor = args.discount_factor  # 다음 state에서의 action value 가치 조정. gamma
        self.epsilon = args.epsilon  # exploration
        self.epsilon_decay = args.epsilon_decay  # exploration decay해줘야 optimal policy로 수렴함.
        self.epsilon_min = args.epsilon_min
        self.model = QNet(state_size, action_size, args.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)  # 학습할 parameter
        # self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True)
        self.tot_episode = args.episodes

    def remember(self, state, action, reward, n_state, done):
        self.memory.append((state, action, reward, n_state, done))  # deque추가

    def choose_action(self, state):  # state를 input으로 받아서 다음 action을 return, epsilon greedy.
        global steps_done
        if np.random.rand() <= self.epsilon:  # epsilon의 확률로 random action
            return random.randrange(self.action_size)  # 범위 내 정수 random
        state = torch.FloatTensor(state).to(device)  # gym에서 state numpy로 반환됨. pytorch model 입력이 tensor여야 해서 변환.
        action_values = self.model(state)
        steps_done += 1
        return torch.argmax(action_values).item()  # action value중 가장 큰 값의 index 반환

    def replay(self, batch_size, episode):
        global steps_done
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([x[0] for x in mini_batch])).to(device)
        actions = torch.LongTensor(np.array([x[1] for x in mini_batch])).to(device)
        rewards = torch.FloatTensor(np.array([x[2] for x in mini_batch])).to(device)
        next_states = torch.FloatTensor(np.array([x[3] for x in mini_batch])).to(device)
        dones = torch.FloatTensor(np.array([x[4] for x in mini_batch])).to(device)

        current_qs = self.model(states)  # vector로 넣으면 각 sample 마다 계산해서 나옴 batch_size x action_size 반환.
        current_qs = current_qs.gather(1, actions.unsqueeze(1)).squeeze(1)  # tensor 차원 맞춰준 후, action뽑아내고 다시 list형태.
        with torch.no_grad():
            next_qs = self.model(next_states).max(1)[0]  # max(1)을 하면 max값, index반환.
        # next_qs = self.model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.discount_factor * next_qs  # done이면 reward만.

        loss = nn.MSELoss()(current_qs, targets)
        # criterion = nn.SmoothL1Loss()
        # loss = criterion(current_qs, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100) #gradient clipping
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            # self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(-(1 - self.epsilon_decay) * episode)
            # self.epsilon = 1.0 - np.exp(-(-config.decay_rate * (episode - self.tot_episode/config.decay_rate2)))
            self.epsilon = args.epsilon_min + (args.epsilon - args.epsilon_min) * np.exp(
                -1. * steps_done / args.epsilon_decay)


def main(args):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNagent(state_size, action_size, args)
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
    args = parser.parse_args()

    wandb.init(project="cartpole", name=args.name, group=args.group, config=vars(args))

    main(args)

# python cartpole_ori.py --episodes 1000 --lr 0.0001 --batch_size 128 --update_period 1 --discount_factor 0.99 --epsilon 0.9 --epsilon_decay 1000 --memory_size 10000 --epsilon_min 0.05 --hidden_size 128 --group test