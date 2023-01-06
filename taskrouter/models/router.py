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
            self.pos_encoding = nn.Parameter(torch.randn(1, 3 + self.config.routing_depth + num_patches, self.args.emb_size))
        else:
            self.pos_encoding = nn.Parameter(torch.randn(1, 3 + self.config.routing_depth, self.args.emb_size))
        
        self.encoder = TransformerEncoder(self.args)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, self.args.emb_size))
        self.layer_norm = nn.LayerNorm(self.args.emb_size)
        self.dropout = nn.Dropout(self.args.drop_rate)

        # Maps each embedding to distr value
        if self.args.vit:
            self.actor_out1 = nn.Linear(self.args.emb_size, self.config.num_modules, bias=False)
        else:
            self.actor_out1 = nn.Linear(self.args.emb_size, self.config.num_modules, bias=False)
            # self.actor_act = nn.GELU()
            # self.actor_out2 = nn.Linear(self.args.emb_size, self.config.num_modules)


    def forward(self, state, action_mask=None):
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = state.shape[0])
        #state = torch.cat((cls_tokens, state), 1)

        # Add position encoding to state

        state_pos = state + self.pos_encoding[:, :state.shape[1]]
        
        

        enc_output = self.encoder(state_pos)

        # Get the class prediction
        #latent_out = enc_output[:, 0]#enc_output.mean(dim=1)
        latent_out = enc_output[:, -1]#(enc_output).mean(dim=1)
        mb_logits = self.actor_out1(self.layer_norm(latent_out))
        # mb_logits = self.actor_act(self.actor_out1(self.layer_norm(latent_out)))
        # mb_logits = self.actor_out2(self.dropout(mb_logits))
        # Apply masks over invalid actions
        if action_mask is not None:
            mb_logits += action_mask * -1e9

        return mb_logits


class Router(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        
        # self._valid_mbs_dict = {}

        self.trans_router = StableTransformer(self.args, self.config)
        self.mb_embs = nn.Embedding(self.config.num_modules, self.args.emb_size)

        self.task_embs = nn.Embedding(self.config.num_tasks, self.args.emb_size)
        #self.apply(self._init_weights)
        
        if self.args.vit:
            self.state_encoder = PatchEncoder(self.args)
        else:
            self.state_encoder = ImageEncoder(self.args, self.config)

    
    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)

    # def configure_router_optimizer(self, args):
    #     """
    #     This long function is unfortunately doing something very simple and is being very defensive:
    #     We are separating out all parameters of the model into two buckets: those that will experience
    #     weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    #     We are then returning the PyTorch optimizer object.
    #     """

    #     # separate out all parameters to those that will and won't experience regularizing weight decay
    #     decay = set()
    #     no_decay = set()
    #     # whitelist_weight_modules = (torch.nn.Linear, )
    #     whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    #     blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    #     for mn, m in self.named_modules():
    #         for pn, p in m.named_parameters():
                
    #             fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                
    #             if fpn in no_decay or fpn in decay:
    #                 continue

    #             if pn.endswith('bias'):
    #                 # all biases will not be decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
    #                 # weights of whitelist modules will be weight decayed
    #                 decay.add(fpn)
    #             elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
    #                 # weights of blacklist modules will NOT be weight decayed
    #                 no_decay.add(fpn)
    #             elif pn.endswith('weight'):
    #                 decay.add(fpn)
    #             else:
    #                 no_decay.add(fpn)

    #     # special case the position embedding parameter in the root GPT module as not decayed
    #     # no_decay.add('pos_emb')
    #     # no_decay.add('global_pos_emb')
    #     # print("decay", decay)
    #     # print("no_decay", no_decay)

    #     # no_decay.add("mb_embs.weight")
    #     # no_decay.add("task_embs.weight")


    #     # validate that we considered every parameter
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     inter_params = decay & no_decay
    #     union_params = decay | no_decay
    #     assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    #     assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
    #                                                 % (str(param_dict.keys() - union_params), )

    #     # create the pytorch optimizer object
    #     optim_groups = [
    #         {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
    #         {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    #     ]
    #     optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=[0.9, 0.95])
    #     return optimizer


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
    
