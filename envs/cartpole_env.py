import gym
import torch
import cv2

import os, sys

file_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(file_path, "../")
sys.path.append(code_path)

from configs import ale_config
import matplotlib.pyplot as plt

from envs.env import Env

class CartpoleEnv(Env):
    def __init__(self, args):
        self.args = args
        self.env = gym.make("CartPole-v1")
    
    def _current_frame(self):
        """Get current frame."""
        # Get frame
        frame = self.env.render(mode='rgb_array')

        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize the frame
        frame = cv2.resize(frame, (self.args.img_size, self.args.img_size), interpolation = cv2.INTER_CUBIC)
        # plt.imshow(frame)
        # plt.show()
        # Normalize frame to [0, 1]
        frame = frame / 255
        
        return torch.tensor(frame, dtype=torch.float32, device=ale_config.device)
    
    def step(self, action):
        _, reward, done, info = self.env.step(action)
        reward  = reward * 0.1
        state = self._current_frame()
        return state, reward, done, info
    
    def reset(self):
        self.env.reset()
        return self._current_frame()
    
    def render(self):
        return self.env.render()