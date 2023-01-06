import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from taskrouter.helpers.gumbel_softmax import gumbel_softmax
from taskrouter.helpers.model_builder import ModelBuilder
from taskrouter.models.blocks import ImageEncoder, PatchEncoder, TransformerEncoder
from taskrouter.models.stable_transformer import StableTransformer

class TransformerRouter(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        # Token for prediction the actor prediction       
        num_patches = (self.config.img_size ** 2) // self.args.patch_size**2
        if self.args.vit:
            self.pos_encoding = nn.Embedding(3 + self.config.routing_depth, self.args.emb_size)
        else:
            self.pos_encoding = nn.Embedding(2 + self.config.routing_depth, self.args.emb_size)
        
        self.encoder = TransformerEncoder(self.args)

        self.layer_norm = nn.LayerNorm(self.args.emb_size)
        self.dropout = nn.Dropout(self.args.drop_rate)
        self.inp_act = nn.Tanh()

        # Maps each embedding to distr value
        if self.args.vit:
            self.actor_out1 = nn.Linear(self.args.emb_size, self.config.num_modules, bias=False)
        else:
            self.actor_out1 = nn.Linear(self.args.emb_size, self.config.num_modules, bias=False)


    def forward(self, state, action_mask=None):
        pos = torch.arange(0, state.shape[1], dtype=torch.long, device=state.device).unsqueeze(0)

        # Add position encoding to state
        state_pos = self.inp_act(state + self.pos_encoding(pos))
        
        enc_output = self.encoder(state_pos)

        # Get the class prediction
        latent_out = enc_output.mean(dim=1)
        mb_logits = self.actor_out1(latent_out)

        # Apply masks over invalid actions
        if action_mask is not None:
            mb_logits += action_mask * -1e9

        return mb_logits


class Router(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        if self.args.use_stable:
            self.trans_router = StableTransformer(self.args, self.config)
        else:
            self.trans_router = TransformerRouter(self.args, self.config)
        self.mb_embs = nn.Embedding(self.config.num_modules, self.args.emb_size)

        self.task_embs = nn.Embedding(self.config.num_tasks, self.args.emb_size)
        
        if self.args.vit:
            self.state_encoder = PatchEncoder(self.args)
        else:
            self.state_encoder = ImageEncoder(self.args, self.config)

    def forward(self, imgs, task_ids, temperature=5, argmax=False):
        batch_size = imgs.shape[0]

        builders = [ModelBuilder(self.config) for _ in range(batch_size)]

        cur_loss = None
        router_outs = torch.zeros(batch_size, self.config.routing_depth, self.config.num_modules, device=self.config.device)
        logits = torch.zeros(batch_size, self.config.routing_depth, self.config.num_modules, device=self.config.device)

        actions = [[] for _ in range(batch_size)]
        
        
        # Mask used for invalid actions
        actions_masks = torch.ones(
            batch_size,
            self.config.routing_depth,
            self.config.num_modules, device=self.config.device)

        # Build the model architecture 
        for i in range(self.config.routing_depth):
            # Update the valid mbs over the batch
            for b_i in range(batch_size): 
                valid_mbs = builders[b_i].get_valid_mbs()
                
                # Update module mask to only mask out invalid actions
                actions_masks[b_i, i, valid_mbs] = 0
            
            # Encode the state
            enc_state = self.state_encoder(imgs)

            # Get task embedding
            task_emb = self.task_embs(task_ids)
            
            # Get the embeddings for the model currently getting constructed
            if i == 0:
                state = torch.cat((task_emb, enc_state), 1)                
            else:
                mb_embs = self.mb_embs(torch.tensor(actions, device=self.config.device))
                state = torch.cat((task_emb, enc_state, mb_embs), 1)

            # Get router prediction
            mb_logits = self.trans_router(
                state, action_mask=actions_masks[:, i])

            logits[:, i] = mb_logits

            # Use Gumbel ST Estimator
            if argmax:
                #mb_choice = mb_logits.argmax(dim=-1)
                mb_choice = torch.zeros_like(mb_logits, device=self.config.device)
                mb_choice[torch.arange(mb_logits.shape[0]), mb_logits.argmax(dim=-1)] = 1
            else:
                mb_choice = F.gumbel_softmax(mb_logits, tau=temperature, hard=True)

            # Iterate over choices in a batch
            for b_i in range(batch_size):
                # Each action needs to be associated with its index in the mb_logits
                action = mb_choice[b_i].nonzero().flatten()[-1].item()
                actions[b_i].append(action)    
                builders[b_i].add_mb(action)
            
            router_outs[:, i] = mb_choice

        return router_outs, logits, actions_masks
    
