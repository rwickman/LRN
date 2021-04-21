import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
from dataclasses import dataclass
from model_util import PASS_FB

class FunctionBlock(nn.Module):
    def __init__(self, layer, norm=None, act=None):
        super().__init__()  
        self.layer = layer
        self.norm = norm
        self.act = act
    
    def forward(self, x):
        x = self.layer(x)
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.act is not None:
            x = self.act(x)

        return x

class MetaWorldFBModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._init_layers()

    def _init_layers(self):
        fbs = []
        for i in range(self.args.num_dup_layers):
            # Input layer
            fbs.append(
                    FunctionBlock(
                        nn.Linear(self.args.mw_obs_size, self.args.hidden_size),
                        nn.LayerNorm(self.args.hidden_size, eps=1e-6), 
                        nn.ReLU()
                    )
                )

            # Hidden layers
            for i in range(self.args.max_depth-2):
                fbs.append(
                    FunctionBlock(
                        nn.Linear(self.args.hidden_size, self.args.hidden_size),
                        nn.LayerNorm(self.args.hidden_size, eps=1e-6),
                        nn.ReLU()
                    )
                )

            # Add last layers
            fbs.append(
                FunctionBlock(
                    nn.Linear(self.args.hidden_size, self.args.mw_action_space * 2)
                )
            )
        
        self._fbs = nn.ModuleList(fbs)
        
    def get_fb(self, idx):
        return self._fbs[idx-1]
    
    def forward(self, x, actions):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            actions: batch of actions giving routing connections.
        
        Returns:
            a Tensor of outputs
        """
        batch_size = x.shape[0]
        actor_actions = torch.zeros((batch_size, self.args.mw_action_space)).cuda()
        actor_actions_logprobs = torch.zeros((batch_size)).cuda()
        for i in range(batch_size):
            cur_x = x[i]
            print("actions", actions)
            for t, a_t in enumerate(actions[i]):
                print("t", t)
                if a_t != PASS_FB:
                    print("a_t", a_t)
                    cur_fb =  self.get_fb(a_t)
                    print(cur_fb)
                    cur_x = cur_fb(cur_x)
            
            
            print("cur_x", cur_x)
            pi_mean = cur_x[:self.args.mw_action_space]
            pi_std =  torch.diag(cur_x[self.args.mw_action_space:])
            print("pi_mean", pi_mean)
            print(pi_std)
            dist = MultivariateNormal(pi_mean, abs(pi_std))
            print(dist.sample())
            actor_actions[i] = dist.sample()
            actor_actions_logprobs[i] = dist.log_prob(actions)

        return actor_actions, actor_actions_logprobs


    def get_num_fbs(self):
        return len(self._fbs)
