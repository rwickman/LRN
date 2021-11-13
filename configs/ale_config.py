from dataclasses import dataclass
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ConvModuleConf:
    shape: tuple
    stride: int


"""Module configs."""
avg_pool_size = 6
conv_config = [
    ConvModuleConf((4, 32), 2),    # (1)
    ConvModuleConf((32, 32), 1),   # (2)
    ConvModuleConf((32, 32), 2),   # (3)
    ConvModuleConf((32, 64), 1),   # (4)
    ConvModuleConf((64, 64), 1),   # (5)
    ConvModuleConf((64, 64), 2),   # (6)
    ConvModuleConf((64, 128), 2),  # (7)
    ConvModuleConf((128, 512), 1), # (8)
]
expand_ratio = 6
#img_size = 84
img_size = 128
num_actions = 2
num_modules = 13

ac_hidden_size = 256

# Create mapping for timestep to valid MBs
valid_mbs_dict = {
    0: set([0]),
    1: set([1, 2]),
    2: set([1, 2]),
    3: set([3]),
    4: set([4, 5]),
    5: set([4, 5]),
    6: set([6]),
    7: set([7]),
    8: set([8]),
    9: set([9]),
    10: set([10]),
    11: set([11]),
    12: set([12]),
}

# valid_mbs_dict = {
#     0: set([0]),
#     1: set([1]),
#     2: set([2]),
#     3: set([3]),
#     4: set([4]),
#     5: set([5]),
#     6: set([6]),
#     7: set([7]),
#     8: set([8]),
#     9: set([9]),
#     10: set([10]),
#     11: set([11]),
# }