from dataclasses import dataclass
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


"""Module configs."""
is_cartpole = False
is_mix = False
forb_lam = 1e-7
if is_cartpole:
    num_envs = 4
    state_size = 4
    num_actions = 2
    envs = ["CartPole-v1"] * num_envs
    reward_scales = [0.1] * num_envs
    envs_ids = torch.tensor([[0], [0], [0], [0]], device=device)
    #envs_ids = torch.arange(num_envs, dtype=torch.int32, device=device).unsqueeze(1)

else:
    state_size = 128
    #num_actions = 6
    num_actions = 6
    if is_mix:
        envs = ["Pong-ram-v0", "Pong-ram-v0", "Qbert-ram-v0", "Qbert-ram-v0", "SpaceInvaders-ram-v0", "SpaceInvaders-ram-v0", "Enduro-ram-v0", "Enduro-ram-v0"]
        #envs = ["Pong-ram-v0", "Pong-ram-v0", "SpaceInvaders-ram-v0", "SpaceInvaders-ram-v0"]
        reward_scales = [1.0, 1.0, 1/25, 1/25, 1/25, 1/25, 1, 1]
        envs_ids = torch.tensor([[0], [0], [1], [1], [2], [2], [3], [3]], device=device)
    else:
        num_envs = 8
        num_actions = 4
        #envs = ["Pong-ram-v0"] * num_envs
        envs = ["Breakout-ram-v0"] * num_envs
        #envs = ["SpaceInvaders-ram-v0"] * num_envs
        #envs = ["Enduro-ram-v0"] * num_envs
        #reward_scales = [1/25] * num_envs
        reward_scales = [1] * num_envs
        #envs_ids = torch.arange(num_envs, dtype=torch.int32, device=device).unsqueeze(1)
        envs_ids = torch.zeros(num_envs, 1, dtype=torch.int32, device=device)

num_modules = 5
ac_hidden_size = 256


fc_hidden_config = [
    (state_size, 128),
    (128, 256),
    (256, 256),
    (256, 256)
]

# Create mapping for timestep to valid MBs
valid_mbs_dict = {
    0: set([0]),
    1: set([1]),
    2: set([2]),
    3: set([3]),
    4: set([4])
}

lstm_timestep = 3

