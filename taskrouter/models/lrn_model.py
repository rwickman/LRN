import torch
import torch.nn as nn


from taskrouter.models.router import Router
from taskrouter.modules.mb_model import ModuleBlockModel


class LRNModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.router_model = Router(args, config).train()
        print("Number of Router parameters: ", sum(p.numel() for p in self.router_model.parameters() if p.requires_grad))

        self.modular_model = ModuleBlockModel(args, config).train()
        print("Number of Modular parameters: ", sum(p.numel() for p in self.modular_model.parameters() if p.requires_grad))
    
    def forward(self, x, task_ids, temperature=5.0, argmax=False):
        router_outs, logits, action_masks = self.router_model(x, task_ids,temperature=temperature, argmax=argmax)
        y_pred = self.modular_model(x, router_outs, task_ids)
        return y_pred, router_outs, logits, action_masks