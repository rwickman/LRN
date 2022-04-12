import torch
from torch.distributions import Categorical
import torch.nn.functional as F

def balance_loss(logits, config,max_prob=0.5, min_prob=0.0901):

    distr = F.softmax(logits, dim=-1).sum(axis=[0,1])/logits.shape[0]
    distr_clipping = distr[distr > max_prob]

    if distr_clipping.shape[0] > 0:
        total_entropy_loss = F.binary_cross_entropy(distr_clipping, torch.tensor([max_prob]).repeat(distr_clipping.shape[0]).to(config.device))
    else:
        total_entropy_loss = 0

    # distr_clipping = distr[distr < min_prob]
    # if distr_clipping.shape[0] > 0:
    #     total_entropy_loss += F.binary_cross_entropy(distr_clipping, torch.tensor([min_prob]).repeat(distr_clipping.shape[0]).to(config.device))

    

    #total_entropy_loss = torch.square(torch.clip(distr - 0.5, min=0)).mean()
    #total_entropy_loss += torch.square(torch.clip(0.01 - distr , min=0)).mean()
    return total_entropy_loss

    






    