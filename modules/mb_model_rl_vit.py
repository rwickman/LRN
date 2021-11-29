import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from model_util import PASS_FB
from modules.rl_modules import ActorCritic
from configs import ale_config_vit as ale_config
from modules.module_block import ModuleBlock, LSTMBlock
from router import ImgEncoder, EncoderLayer
from einops import rearrange, repeat

class ModuleBlockModelRLViT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        
        # Token for prediction the actor prediction and critic
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.args.hidden_size))
        
        # Create learned positional encodings
        num_patches = (self.args.img_size**2) // self.args.patch_size**2
        max_inp = 1 + num_patches
        self.dropout = nn.Dropout(self.args.drop_rate)
        self.pos_enc = nn.Parameter(torch.randn(1, max_inp, self.args.hidden_size))
        self.lstm_blocks_idxs = []
        self._init_layers()
        self._num_timesteps = 0
        #self.create_mb_probs()

    def _init_layers(self):
        modules = []
        for i in range(self.args.num_dup_layers):
            """Input layers."""
            # (1)
            modules.append(
                ModuleBlock(ImgEncoder(self.args)))
            
            """Hidden Layers."""
            # (2 - num_enc_layers)
            for i in range(1, ale_config.num_enc_layers+1):
                modules.append(EncoderLayer(self.args))

            # (num_enc_layers + 1)
            modules.append(
                LSTMBlock(
                    nn.LSTMCell(ale_config.hidden_size, ale_config.hidden_size)
                )
            )
            self.lstm_blocks_idxs.append(len(modules) - 1)

            """Actor-critic Layers"""
            # (num_enc_layers + 2)
            modules.append(ActorCritic(ale_config))
        
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

    def add_token(self, state):
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = state.shape[0])
        state = torch.cat((cls_tokens, state), 1)
        state = state * torch.sqrt(torch.tensor(self.args.hidden_size, dtype=torch.float32, device=ale_config.device))
        state = state + self.pos_enc[:, :state.shape[1], :]
        state = self.dropout(state)
        return state

    def forward(self, x, actions, step_to_batch_idx=None, return_hidden_state=False):
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
                if t == 0:
                    # Run through image encoder
                    cur_x = cur_mb(cur_x)
                    cur_x = self.add_token(cur_x)
                elif t >= 1 and t <= ale_config.num_enc_layers:
                    # Run through encoder layer
                    cur_x, _ = cur_mb(cur_x) 
                
                elif t == ale_config.num_enc_layers + 1:
                    # Get prediction token
                    cur_x = cur_x[:, 0]

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
                            

                        cur_x = cur_mb(cur_x)
                    else:
                        cur_x = cur_mb(cur_x, self._num_timesteps)
                        self._num_timesteps += 1
                    
                    if return_hidden_state:
                        hidden_state = cur_x.clone()


                else:
                    # Run through ActorCritic module
                    actor_out, critic_out = cur_mb(cur_x)

            actor_preds[i] = actor_out
            critic_preds[i] = critic_out
        
        if return_hidden_state:
            return actor_preds, critic_preds, hidden_state
        else:
            return actor_preds, critic_preds

    def get_num_mbs(self):
        return len(self._mbs)
