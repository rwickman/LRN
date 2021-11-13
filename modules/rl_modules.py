import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(oup),
            )

    def forward(self, x):
        #print(x.shape)
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
