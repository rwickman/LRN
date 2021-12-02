import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from model_util import PASS_FB
from modules.rl_modules import *
from configs import ale_config
from modules.rl_modules import ActorCritic, InvertedResidualBlock
from modules.module_block import ModuleBlock, LSTMBlock

class ModuleBlockModelRL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.lstm_blocks_idxs = []
        self._init_layers()
        self._num_timesteps = 0

    def _init_layers(self):
        modules = []
        for i in range(self.args.num_dup_layers):
            """Input layers."""
            # (1)
            for in_dim, out_dim in ale_config.fc_hidden_config:
                modules.append(
                    ModuleBlock(
                        nn.Linear(in_dim, out_dim),
                        nn.LayerNorm(out_dim),
                        nn.GELU()
                    )
                )
            
            """Hidden Layers."""

            # modules.append(
            #     LSTMBlock(
            #         nn.LSTMCell(
            #             ale_config.fc_hidden_config[-1][1],
            #             ale_config.fc_hidden_config[-1][1])
            #     )
            # )
            # self.lstm_blocks_idxs.append(len(modules) - 1)
            

            """Actor-critic Layers"""
            # Actor layers
            # (10)
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
    

    # def save_states(self, should_save):
    #     for i in self.lstm_blocks_idxs:
    #         self._mbs[i].save_states = should_save

    # def setup_next_episode(self):
    #     """Add initial hidden/cells states to LSTM modules."""
    #     for i in self.lstm_blocks_idxs:
    #         self._mbs[i].add_zero()
        

    # def reset(self):
    #     self._num_timesteps = 0
    #     for i in self.lstm_blocks_idxs:
    #         self._mbs[i].reset()
    
    def get_mb_num(self, action):
        return (action - 1) % 6

    def get_mb(self, idx):
        return self._mbs[idx]
    

    def forward(self, x, actions):
        """Forward the batch of inputs through the Function Blocks.

        Args:
            x: batch of inputs
            actions: batch of actions giving routing connections.
        
        Returns:
            a Tensor of outputs
        """
        # TODO: optimize this to predict over a batch at a time
        # TODO: This will require iterating over grouping together inputs each timestep
        batch_size = x.shape[0]
        actor_preds = torch.zeros((batch_size, ale_config.num_actions), device=ale_config.device).detach()
        #critic_preds = torch.zeros((batch_size, 1), device=ale_config.device).detach()
        # hidden_states = []
        for i in range(batch_size):
            cur_x = x[i:i+1]
            
            for t, a_t in enumerate(actions[i]):
                cur_mb = self.get_mb(a_t)
                
                # if t == ale_config.lstm_timestep:
                #     # Run through LSTM
                #     if step_to_batch_idx is not None:
                #         if i in step_to_batch_idx:
                #             # Make LSTM module set initial state to saved state
                #             cur_mb.init_states(step_to_batch_idx[i])

                #         cur_x = cur_mb(cur_x, batch_idxs[i])
                #     else:
                #         cur_x = cur_mb(cur_x, self._num_timesteps)
                #         self._num_timesteps += 1
                    
                #     if return_hidden_state:
                #         hidden_states.append(cur_x.clone())


                if t >= ale_config.lstm_timestep + 1:
                    # Run through actor layers
                    actor_out = cur_mb(cur_x)
                else:
                    # Run through other modules, including critic modules
                    cur_x = cur_mb(cur_x)

            actor_preds[i] = actor_out
            # critic_preds[i] = critic_out 
        
        #if return_hidden_state:
        return actor_preds#, critic_preds#, torch.cat(hidden_states)
        # else:
        #     return actor_preds, critic_preds

    def get_num_mbs(self):
        return len(self._mbs)
