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
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                nn.InstanceNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                nn.InstanceNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=True),
                nn.InstanceNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
                nn.InstanceNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=True),
                nn.InstanceNorm2d(oup),
            )

    def forward(self, x):
        #print(x.shape)
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.inv_res_1 = InvertedResidual(inp, oup, 2, expand_ratio)
        self.inv_res_2 = InvertedResidual(inp, oup, 1, expand_ratio)
        self.inv_res_3 = InvertedResidual(inp, oup*2, stride, expand_ratio)
    
    def forward(self, x):
        x = self.inv_res_1(x)
        x = self.inv_res_2(x)
        x = self.inv_res_3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.actor_1 = nn.Linear(self.args.ac_hidden_size, self.args.ac_hidden_size)
        self.actor_2 = nn.Linear(self.args.ac_hidden_size, self.args.num_actions)
        self.actor_act_1 = nn.GELU()
        self.actor_act_2 = nn.Softmax(dim=-1)
        
        self.critic_1 = nn.Linear(self.args.ac_hidden_size, self.args.ac_hidden_size)
        self.critic_2 = nn.Linear(self.args.ac_hidden_size, 1)
        self.critic_act_1 = nn.GELU()

    def forward(self, x):
        actor_out = self.actor_act_1(self.actor_1(x))
        actor_out = self.actor_act_2(self.actor_2(actor_out))

        critic_out = self.critic_act_1(self.critic_1(x))
        critic_out = self.critic_2(critic_out)

        return actor_out, critic_out

class ResidualBlock(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.conv_1 = nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding="same")
        self.act_1 = nn.GELU()
        self.conv_2 = nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1,padding="same")
        self.act_2 = nn.GELU()
    
    def forward(self, x):
        x_out = self.act_1(x)
        x_out = self.act_2(self.conv_1(x_out))
        x_out = self.conv_2(x_out)
        return x + x_out

class ImpalaBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block_1 = ResidualBlock(out_ch)
        self.res_block_2 = ResidualBlock(out_ch)
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        return x



