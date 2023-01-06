import torch
import torch.nn as nn 
import torchvision.models as torch_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def src_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        enc_layer = nn.TransformerEncoderLayer(
                d_model=self.args.emb_size,
                dim_feedforward=self.args.dff,
                activation="gelu",
                nhead=self.args.num_heads,
                dropout=self.args.drop_rate,
                batch_first=True,
                norm_first=True)

        self.enc_1 = nn.TransformerEncoder(enc_layer, self.args.num_enc_layers)


    def forward(self, x):
        """
            x: task emb, example emb, used mbs embs
        """
        mask = src_mask(x.shape[1]).to("cuda")
        mask[0, 1] = 0
        x = self.enc_1(x, mask)
        
        return x

class PatchEncoder(nn.Module):
    """Create patch embeddings for a given image."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_dim = self.args.patch_size ** 2 * self.args.in_channels
        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(1, 32, 3, padding="same"),
            # nn.BatchNorm2d(32), 
            # nn.GELU(),
            # nn.Conv2d(32, 1, 3, padding="same"),
            # nn.BatchNorm2d(1), 
            # nn.GELU(),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.args.patch_size, p2 = self.args.patch_size),
            nn.Linear(self.patch_dim, self.args.emb_size),
        )

    def forward(self, imgs):
        # Extract patches and embed them
        patches = self.to_patch_embedding(imgs)
        return patches

class ImageEncoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        

        # self.model = nn.Sequential(
        #     nn.Conv2d(self.config.in_channels, 64, 8, stride=4, padding=0),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, stride=2, padding=0),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(128, 512, 3, stride=1, padding=0),
        #     nn.LeakyReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(512, self.args.emb_size),
        #     nn.Tanh()
        # )
        self.model = torch_models.mobilenet_v3_small(True)
        if self.config.in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.args.emb_size)

    def forward(self, x):
        x = self.model(x)
        return x.unsqueeze(1)