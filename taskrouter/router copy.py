import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from gumbel_softmax import gumbel_softmax
from model_builder import ModelBuilder



class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        enc_layer = nn.TransformerEncoderLayer(
                d_model=self.args.emb_size,
                dim_feedforward=self.args.dff,
                activation="gelu",
                nhead=self.args.num_heads,
                dropout=self.args.drop_rate,
                batch_first=True)

        self.enc = nn.TransformerEncoder(enc_layer, self.args.num_enc_layers)

    def forward(self, x):
        """
            x: task emb, example emb, used mbs embs
        """
        x = self.enc(x)
        
        return x



class RouterImgEncoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.model = torch_models.mobilenet_v3_small(False)
        self.model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.args.emb_size)

    def forward(self, x):
        x = self.model(x)
        print("x.shape", x.shape)
        return x.zz


class TransformerRouter(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        # Token for prediction the actor prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.args.emb_size))

        
        num_patches = (self.config.img_size ** 2) // self.args.patch_size**2
        self.pos_encoding = nn.Parameter(torch.randn(1, 3 + self.config.routing_depth, self.args.emb_size))
        
        self.encoder = Encoder(self.args)
        
        # Maps each embedding to distr value
        self.actor_out1 = nn.Linear(self.args.emb_size, self.args.emb_size)
        self.actor_out2 = nn.Linear(self.args.emb_size, self.config.num_modules)

    def forward(self, state, action_mask=None):
       
        # Add prediction token to state
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = state.shape[0])
        state = torch.cat((cls_tokens, state), 1)
        state = state + self.pos_encoding[:, :state.shape[1]]

        
        enc_output = self.encoder(state)
        

        # Get the class prediction
        mb_logits = F.gelu(self.actor_out1(enc_output[:, 0]))
        mb_logits = self.actor_out2(mb_logits)


        # Apply masks over invalid actions
        if action_mask is not None:
            mb_logits += action_mask * -1e9

        # Get probabilities over modules
        
        return mb_logits

class PatchEncoder(nn.Module):
    """Create patch embeddings for a given image."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_dim = self.args.patch_size ** 2 * self.args.in_channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.args.patch_size, p2 = self.args.patch_size),
            nn.Linear(self.patch_dim, self.args.emb_size),
        )

    def forward(self, imgs):
        # Extract patches and embed them
        patches = self.to_patch_embedding(imgs)
        return patches

class Router(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        # self._valid_mbs_dict = {}

        self.trans_router = TransformerRouter(self.args, self.config)
        print("self.config.num_modules", self.config.num_modules)
        self.mb_embs = nn.Embedding(self.config.num_modules, self.args.emb_size)

        self.task_embs = nn.Embedding(self.config.num_tasks, self.args.emb_size)
        
        self.state_encoder = RouterImgEncoder(self.args, self.config)


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
                mb_choice = gumbel_softmax(mb_logits, self.config.device, temperature=0.00001)
            else:
                mb_choice = gumbel_softmax(mb_logits, self.config.device, temperature)

            # Iterate over choices in a batch
            for b_i in range(batch_size):
                # Each action needs to be associated with its index in the mb_logits
                action = mb_choice[b_i].nonzero().flatten()[-1].item()
                actions[b_i].append(action)    
                builders[b_i].add_mb(action)
            
            router_outs[:, i] = mb_choice

        return router_outs, logits
    
