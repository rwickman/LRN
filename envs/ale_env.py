import gym
from gym.wrappers import AtariPreprocessing

from envs.env import Env


class ALEEnv(Env):
    def __init__(self, args):
        self.args = args
        self.env = AtariPreprocessing(gym.make('PongNoFrameskip-v4'), frame_skip=4, scale_obs=True)
        
