from abc import ABC

import torch

class Env(ABC):
    """A wrapper around a Gym environment."""
    def __init__(self, env):
        self.env = env

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return torch.tensor(state).cuda(), reward, done, info
    
    def reset(self):
        return torch.tensor(self.env.reset()).cuda()

    
    def render(self):
        return self.env.render()