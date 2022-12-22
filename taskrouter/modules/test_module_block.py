import torch
from mb_model import ModuleBlockModel

class TempConfig:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_sizes=[4, 4, 2]
    img_size=105
    in_channels=3
    num_filters=32
    num_hidden=1
    num_downsample=1

config = TempConfig()

mb_model = ModuleBlockModel(config).to(config.device)

fake_model_out = torch.zeros(1, 4, 14, device=config.device)
fake_model_out[0, 0, 0] = 1
fake_model_out[0, 1, 5] = 1
fake_model_out[0, 2, 3+6 +1] = 1
fake_model_out[0, 3, 3+6 +3 +1] = 1
x = torch.randn(1, 3, 105, 105, device=config.device)
print("fake_model_out", fake_model_out)
y = mb_model(x, fake_model_out)



print(y)