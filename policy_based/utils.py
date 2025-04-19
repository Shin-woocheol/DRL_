import random
import numpy as np
import torch
import gymnasium as gym
import argparse
from collections import deque


def fix_seed(seed = 777):
    random.seed(seed) # python random
    np.random.seed(seed) # numpy random
    torch.manual_seed(seed) # pytorch random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) #single GPU
        torch.cuda.manual_seed_all(seed) #multi GPU
        torch.backends.cudnn.deterministic = True #cudnn non deterministic off
        torch.backends.cudnn.benchmark = False #cudnn 성능 최적화 off

def evaluate(env_name, agent, seed, eval_iteration, action_type, visualize = False):
    #현재까지 학습된 policy에 대해서, max prob을 가지는 action만을 뽑아서 평가시작.
    env = gym.make(env_name)
    scores = []
    for i in range(eval_iteration): # evaluation을 위해 작동할 episodes 수
        state, _ = env.reset(seed = seed + 100 + i) # 각 evaluation마다 시작 state다를 수 있음.
        terminated, truncated, score = False, False, 0
        while not (terminated or truncated): #한 episode시작.
            action = agent.act(state, training=False)
            if action_type == 'discrete':
                n_state, reward, terminated, truncated, _ = env.step(action[0]) # 아마 numpy array주는 것이어서.
            elif action_type == 'continuous':
                n_state, reward, terminated, truncated, _ = env.step(action * 2.0) # pendulum의 action sapce는 -2 to 2
            score += reward
            state = n_state
        scores.append(score)
        if i == eval_iteration -2:
            env = gym.make(env_name, render_mode = "human" if visualize else None)
    env.close()
    return round(np.mean(scores), 4) # eval time동안 얻은 return 평균내기.

def parse_hidden_dims(s):
    try:
        return tuple(map(int, s.strip("()").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("hidden dim input syntax error")

class RolloutBuffer:
    '''
    general buffer.
    '''
    def __init__(self):
        self.buffer = list()

    def store(self, trainsition):
        self.buffer.append(trainsition)
    
    def sample(self):
        state, action, reward, n_state, terminated, truncated = map(np.array, zip(*self.buffer))
        self.buffer.clear()
        state, action, reward, n_state, terminated, truncated = map(
            lambda x: torch.as_tensor(x, dtype=torch.float),
            [state, action, reward, n_state, terminated, truncated]
        )
        reward = reward.unsqueeze(1)
        done = (truncated + terminated) > 0
        done = done.to(torch.float)
        done = done.unsqueeze(1)

        return state, action, reward, n_state, done

    @property # @property decorator를 붙이면 buffer.size 로 buffer.size()를 이용할 수 있다.
    def size(self):
        return len(self.buffer)

class NStepRolloutBuffer(RolloutBuffer):
    def __init__(self, n_step, gamma):
        super().__init__()
        self.n_step = n_step
        self.gammas = [gamma ** t for t in range(n_step)]
        self.n_step_tracker = deque(maxlen=self.n_step) # n_step 계산을 위해서 저장. n_step넘어가면 FIFO로 삭제됨.

    def _get_n_step_transition(self): # implicit private method
        '''
        n_step return : r_t + gamma * r_t+1 + ... + gamma**n-1 * r_t+n-1 + V(s_t+n) 
        state value function 이전까지의 값을 return 
        return 받은 G + state function(n_state)를 해주면 n_step target이 됨.
        '''
        G = 0
        state, action, _, _, _, _ = self.n_step_tracker[0]
        for t in range(self.n_step):
            _, _, reward, n_state, terminated, truncated = self.n_step_tracker[t]
            G += self.gammas[t] * reward
            if terminated or truncated:
                break

        return (state, action, G, n_state, terminated, truncated)
        
    def store(self, transition):
        self.n_step_tracker.append(transition)
        if len(self.n_step_tracker) == self.n_step:
            n_step_trainsition = self._get_n_step_transition()
            super().store(n_step_trainsition)

    def sample(self): #이걸 할 때는 update이므로 buffer를 전부 clear해줌.
        transitions = super().sample()
        self.n_step_tracker.clear()
        return transitions