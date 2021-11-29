import torch

#from configs import ale_config_vit as ale_config
from configs import ale_config

class RLModelBuilder:
    """Controls what modules can be used each timestep."""
    def __init__(self, args):
        self.args = args
        self._prev_mb_idxs = []
    
    def attempt_add_mb(self, mb_idx):
        # Set to the base index
        #self._prev_mb_idx = mb_idx % ale_config.num_modules
        self._prev_mb_idxs.append(mb_idx)
        return True

    def get_valid_mbs(self, cur_depth):
        # Get what modules are available for the current depth
        valid_mbs_set = ale_config.valid_mbs_dict[cur_depth].copy()

        # Add duplicates
        valid_mbs = torch.tensor(list(valid_mbs_set)).unsqueeze(0)
        valid_mbs = valid_mbs.repeat(self.args.num_dup_layers, 1)
        valid_mbs = valid_mbs + torch.arange(self.args.num_dup_layers).unsqueeze(1) * ale_config.num_modules
        valid_mbs = valid_mbs.flatten()
        valid_mbs = valid_mbs.tolist()

        # Remove already used MBs
        for idx in self._prev_mb_idxs:
            if idx in valid_mbs:
                valid_mbs.remove(idx)


        return valid_mbs 
