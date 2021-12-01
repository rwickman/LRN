import gym
import torch

from envs.env import Env
from configs import ale_config

class ALEEnv(Env):
    def __init__(self, args, env_name="Pong-ram-v0", reward_scale=1.0):
        self.args = args
        #self.env = gym.make('Pong-ram-v0')
        self.env = gym.make(env_name)
        self.reward_scale = reward_scale

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = torch.tensor(state, dtype=torch.float32)
        reward = reward * self.reward_scale
        state = state/255
        return state, reward, done, info
    
    def reset(self):
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        state = state/255
        return state#self._current_frame()
    
    def render(self):
        return self.env.render()