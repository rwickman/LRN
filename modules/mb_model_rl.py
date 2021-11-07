import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from model_util import PASS_FB
from modules.rl_modules import *
from configs import ale_config
from modules.module_block import ModuleBlock

class ModuleBlockModelRL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        
        self.avg_pool = nn.AvgPool2d(ale_config.avg_pool_size)
        self._init_layers()
        #self.create_fb_probs()

    def _init_layers(self):
        modules = []
        for i in range(self.args.num_dup_layers):
            """Input layers."""
            # (1)
            modules.append(
                    ModuleBlock(
                        nn.Conv2d(ale_config.conv_config[0].shape[0], ale_config.conv_config[0].shape[1], 3, ale_config.conv_config[0].stride, 1, bias=False),
                        nn.BatchNorm2d(ale_config.conv_config[0].shape[1]), 
                        nn.ReLU()))
            
            """Hidden Layers."""
            # (2 - 7)
            for i in range(1, 7):
                modules.append(
                        ModuleBlock(
                            InvertedResidual(
                                ale_config.conv_config[i].shape[0],
                                ale_config.conv_config[i].shape[1],
                                ale_config.conv_config[i].stride,
                                ale_config.expand_ratio)))
            # (8)
            modules.append(
                ModuleBlock(
                    nn.Conv2d(ale_config.conv_config[7].shape[0], ale_config.conv_config[7].shape[1], 1, stride=ale_config.conv_config[7].stride, padding=0),
                    nn.BatchNorm2d(ale_config.conv_config[7].shape[1]),
                    nn.LeakyReLU(0.2)))


            """Actor-critic Layers"""
            # Actor layers
            # (9)
            modules.append(
                ModuleBlock(
                    nn.Linear(ale_config.conv_config[7].shape[1], ale_config.ac_hidden_size),
                    None,
                    nn.LeakyReLU(0.2)
                )
            )

            # (10)
            modules.append(
                ModuleBlock(
                    nn.Linear(ale_config.ac_hidden_size, ale_config.num_actions),
                    None,
                    nn.Softmax(dim=-1)
                )
            )

            # Critic layers
            # (11)
            modules.append(
                ModuleBlock(
                    nn.Linear(ale_config.conv_config[7].shape[1], ale_config.ac_hidden_size),
                    None,
                    nn.LeakyReLU(0.2)
                )
            )

            # (12)
            modules.append(
                ModuleBlock(
                    nn.Linear(ale_config.ac_hidden_size, 1)
                )
            )
        
        self._mbs = nn.ModuleList(modules)
        

    def action_to_fb_type(self, action):
        if action == PASS_FB:
            return "E"
        else:
            return "E" + str((action-1) % 6)
    
    def get_fb_num(self, action):
        return (action - 1) % 6

    def get_fb(self, idx):
        return self._mbs[idx]
    
    def forward(self, x, actions):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            actions: batch of actions giving routing connections.
        
        Returns:
            a Tensor of outputs
        """
        batch_size = x.shape[0]
        actor_preds = torch.zeros((batch_size, ale_config.num_actions), device=ale_config.device)
        critic_preds = torch.zeros((batch_size, 1), device=ale_config.device)
        for i in range(batch_size):
            cur_x = x[i:i+1]
            for t, a_t in enumerate(actions[i]):

                print("t, cur_x.shape", t, cur_x.shape)
                if t == 8:
                    # Perform average-pooling
                    cur_x = self.avg_pool(cur_x)
                    cur_x = cur_x.view(cur_x.shape[0], -1)
                
                
                if t >= 8 and t <= 9:
                    # Run through actor layers
                    print(t)
                    if t == 8:
                        actor_out = cur_x
                    
                    cur_fb =  self.get_fb(a_t)
                    actor_out = cur_fb(actor_out)
                else:
                    # Run through other modules, including critic modules
                    cur_fb = self.get_fb(a_t)
                    cur_x = cur_fb(cur_x)

            actor_preds[i] = actor_out
            critic_preds[i] = cur_x

        return actor_preds, critic_preds

    def get_num_modules(self):
        return len(self._mbs)
