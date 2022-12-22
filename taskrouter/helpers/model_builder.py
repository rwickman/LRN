import torch.nn as nn 

class ModelBuilder:
    """Controls what modules can be used each timestep."""
    def __init__(self, config):
        self.config = config
        self._prev_mb_idxs = []
    
    def add_mb(self, mb_idx):
        # Set to the base index
        #self._prev_mb_idx = mb_idx % ale_config.num_modules
        self._prev_mb_idxs.append(mb_idx)
        return True

    def get_valid_mbs(self):
        # Get what modules are available for the current depth
        valid_mbs_set = set(self.config.valid_mbs[len(self._prev_mb_idxs)])
        prev_mbs = set(self._prev_mb_idxs)
        valid_mbs_set = valid_mbs_set.difference(prev_mbs)
        
    
        return list(valid_mbs_set) 
