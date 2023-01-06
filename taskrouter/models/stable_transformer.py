import torch
import torch.nn as nn
import torch.nn.functional as F
from taskrouter.helpers.model_util import src_mask

class GatingMechanism(nn.Module):
    def __init__(self, d_input, bg=0.1):
        super().__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g

class Feedforward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(args.emb_size, args.dff),
            nn.GELU(),
            nn.Dropout(args.drop_rate),
            nn.Linear(args.dff, args.emb_size),
            nn.Dropout(args.drop_rate)
        )
    def forward(self, x):
        return self.ff(x)

        

class StableTransformerLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.gate1 = GatingMechanism(args.emb_size, bg=0)
        self.gate2 = GatingMechanism(args.emb_size, bg=0)
        self.mha = nn.MultiheadAttention(
            args.emb_size, args.num_heads, args.drop_rate, batch_first=True)
        self.ff = Feedforward(args)
        self.act = nn.ReLU()
        self.norm1 = nn.LayerNorm(args.emb_size)
        self.norm2 = nn.LayerNorm(args.emb_size)

    def forward(self, E):
        #print("E.shape", E.shape)
        y_bar = self.norm1(E)
        if self.args.decoder:
            mask = src_mask(E.shape[1]).to(y_bar.device.type)
            mask[0, 1] = 0
        else:
            mask = None

        y_bar, att = self.mha(y_bar, y_bar, y_bar, attn_mask=mask)
        # print("att", att.shape, att[0:2])
        y = self.gate1(E, self.act(y_bar))
        E_bar = self.ff(self.norm2(y))
        E_next = self.gate2(y, self.act(E_bar))
        return E_next

class StableTransformer(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        #self.pos_encoding = nn.Embedding()
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 2 + self.config.routing_depth, self.args.emb_size))
        self.encoder = nn.Sequential(
            *[StableTransformerLayer(self.args) for _ in range(self.args.num_enc_layers)])

        self.actor_out1 = nn.Linear(self.args.emb_size, self.config.num_modules, bias=False)
    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.normal_(mean=0.0, std=0.001)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.normal_(mean=0.0, std=0.001)
            #module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, state, action_mask=None, mask=None):
        # state = torch.tanh(state)
        

        # print("IMG STATS", state[:, 0].min(), state[:, 0].max(), state[:, 0].mean())
        # print("TASK EMB STATS", state[:, 1].min(), state[:, 1].max(), state[:, 1].mean())

        # print("POS EMB STATS", self.pos_encoding.min(), self.pos_encoding.max(), self.pos_encoding.mean())
        # print("self.pos_encoding[:, 1, 0:10]", self.pos_encoding[:, 1, 0:10])
        # Add position encoding to state
        state_pos = torch.tanh(state + self.pos_encoding[:, :state.shape[1]])
        # state_pos = state + self.pos_encoding[:, :state.shape[1]] * 0.0001

 



        # Get the MB logits
        enc_output = self.encoder(state_pos)
        if self.args.decoder:
            latent_out = enc_output[:, -1]
        else:
            latent_out = enc_output.mean(dim=1)

        mb_logits = self.actor_out1(latent_out)
        
        # Apply masks over invalid actions
        if action_mask is not None:
            mb_logits += action_mask * -1e9
        # print("mb_logits", mb_logits[0])
        return mb_logits
