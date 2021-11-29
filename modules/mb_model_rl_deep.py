import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from model_util import PASS_FB, conv2d_out
from modules.rl_modules import *
from configs import ale_config
from modules.rl_modules import ActorCritic, ImpalaBlock
from modules.module_block import ModuleBlock, LSTMBlock

class ModuleBlockModelRL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        
        #self.avg_pool = nn.AvgPool2d(ale_config.avg_pool_size)
        self.lstm_blocks_idxs = []
        self._init_layers()
        self._num_timesteps = 0
        #self.create_mb_probs()

    def _init_layers(self):
        modules = []
        for i in range(self.args.num_dup_layers):
            # """Input layers."""
            # # (1)
            # modules.append(
            #         ModuleBlock(
            #             nn.Conv2d(ale_config.conv_config[0].shape[0], ale_config.conv_config[0].shape[1], 3, ale_config.conv_config[0].stride, 1, bias=False),
            #             nn.InstanceNorm2d(ale_config.conv_config[0].shape[1]), 
            #             nn.ReLU()))
            
            # """Hidden Layers."""
            # # (2 - 7)
            # for i in range(1, 7):
            #     modules.append(
            #             ModuleBlock(
            #                 InvertedResidual(
            #                     ale_config.conv_config[i].shape[0],
            #                     ale_config.conv_config[i].shape[1],
            #                     ale_config.conv_config[i].stride,
            #                     ale_config.expand_ratio)))
            # # (8)
            # modules.append(
            #     ModuleBlock(
            #         nn.Conv2d(ale_config.conv_config[7].shape[0], ale_config.conv_config[7].shape[1], 1, stride=ale_config.conv_config[7].stride, padding=0),
            #         nn.InstanceNorm2d(ale_config.conv_config[7].shape[1]),
            #         nn.GELU()))
            # (1)
            modules.append(
                ModuleBlock(
                    ImpalaBlock(self.args.in_channels, 16),
                    nn.InstanceNorm2d(16)
            ))
            # (2)
            modules.append(
                ModuleBlock(
                    ImpalaBlock(16, 32),
                    nn.InstanceNorm2d(32)
            ))
            # (3)
            modules.append(
                ModuleBlock(
                    ImpalaBlock(32, 32),
                    nn.InstanceNorm2d(32),
                    nn.GELU()
            ))


            # modules.append(ModuleBlock(
            #     nn.Conv2d(16, 32, kernel_size=4, stride=2)
            # ))
            # (4)
            fc_in = 8192# int(conv2d_out(conv2d_out(self.args.img_size, 8, 4), 4, 2) * 2 * 32)

            # print("conv2d_out(self.args.img_size, 8, 4)", conv2d_out(self.args.img_size, 8, 4))
            # print("fc_in", fc_in)
            modules.append(
                ModuleBlock(
                    nn.Linear(fc_in, 256),
                    nn.LayerNorm(256),
                    nn.GELU()
                ))

            # (5)
            modules.append(
                LSTMBlock(
                    nn.LSTMCell(256, 256)
                )
            )
            
            self.lstm_blocks_idxs.append(len(modules) - 1)
            

            """Actor-critic Layers"""
            # Actor layers
            # (6)
            modules.append(ActorCritic(ale_config))
            # modules.append(
            #     ModuleBlock(
            #         nn.Linear(ale_config.conv_config[7].shape[1], ale_config.ac_hidden_size),
            #         nn.LayerNorm(ale_config.ac_hidden_size),
            #         nn.GELU()
            #     )
            # )
            

            # # (11)
            # modules.append(
            #     ModuleBlock(
            #         nn.Linear(ale_config.ac_hidden_size, ale_config.num_actions),
            #         None,
            #         nn.Softmax(dim=-1)
            #     )
            # )

            # # Critic layers
            # # (12)
            # modules.append(
            #     ModuleBlock(
            #         nn.Linear(ale_config.conv_config[7].shape[1], ale_config.ac_hidden_size),
            #         nn.LayerNorm(ale_config.ac_hidden_size),
            #         nn.GELU()
            #     )
            # )

            # # (13)
            # modules.append(
            #     ModuleBlock(
            #         nn.Linear(ale_config.ac_hidden_size, 1)
            #     )
            # )
        
        self._mbs = nn.ModuleList(modules)
    

    def save_states(self, should_save):
        for i in self.lstm_blocks_idxs:
            self._mbs[i].save_states = should_save

    def setup_next_episode(self):
        """Add initial hidden/cells states to LSTM modules."""
        for i in self.lstm_blocks_idxs:
            self._mbs[i].add_zero()
        

    def reset(self):
        self._num_timesteps = 0
        for i in self.lstm_blocks_idxs:
            self._mbs[i].reset()
    
    def get_mb_num(self, action):
        return (action - 1) % 6

    def get_mb(self, idx):
        return self._mbs[idx]
    

    def forward(self, x, actions, step_to_batch_idx=None, return_hidden_state=False, batch_idxs=[]):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            actions: batch of actions giving routing connections.
        
        Returns:
            a Tensor of outputs
        """
        # TODO: optimize this to predict over a batch at a time
        # TODO: This will require iterating over grouping together inputs each timestep
        import time
        batch_size = x.shape[0]
        actor_preds = torch.zeros((batch_size, ale_config.num_actions), device=ale_config.device).detach()
        critic_preds = torch.zeros((batch_size, 1), device=ale_config.device).detach()
        hidden_state = None
        for i in range(batch_size):
            cur_x = x[i:i+1]
            
            for t, a_t in enumerate(actions[i]):
                cur_mb = self.get_mb(a_t)
                
                if t == ale_config.lstm_timestep:
                    # Perform average-pooling
                    # cur_x = self.avg_pool(cur_x)
                    

                    # Run through LSTM
                    if step_to_batch_idx is not None:
                        if i in step_to_batch_idx:
                            # Make LSTM module set initial state to saved state
                            # print("Initing STATES")
                            # time.sleep(1)
                            cur_mb.init_states(step_to_batch_idx[i])
                        # else:
                        #     print("NOT Initing STATES")
                        #     time.sleep(3)

                        cur_x = cur_mb(cur_x, batch_idxs[i])
                    else:
                        cur_x = cur_mb(cur_x, self._num_timesteps)
                        self._num_timesteps += 1
                    
                    if return_hidden_state:
                        hidden_state = cur_x.clone()


                elif t >= ale_config.lstm_timestep + 1:
                    # Run through actor layers
                    actor_out, critic_out = cur_mb(cur_x)
                else:
                    # print("cur_x.shape", cur_x.shape)
                    if t == ale_config.lstm_timestep - 1:
                        cur_x = cur_x.view(cur_x.shape[0], -1)
                        #print("cur_x.shape", cur_x.shape)
                    # Run through other modules, including critic modules
                    cur_x = cur_mb(cur_x)

            actor_preds[i] = actor_out
            critic_preds[i] = critic_out 
        
        if return_hidden_state:
            return actor_preds, critic_preds, hidden_state
        else:
            return actor_preds, critic_preds

    def get_num_mbs(self):
        return len(self._mbs)
