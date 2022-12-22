import torch.nn as nn 
import torchvision.models as torch_models

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
                batch_first=True)

        self.enc_1 = nn.TransformerEncoder(enc_layer, self.args.num_enc_layers)


    def forward(self, x):
        """
            x: task emb, example emb, used mbs embs
        """
        x = self.enc_1(x)
        
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
        self.model = torch_models.mobilenet_v3_small(True)
        self.model.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, self.args.emb_size)

    def forward(self, x):
        x = self.model(x)
        return x.unsqueeze(1)