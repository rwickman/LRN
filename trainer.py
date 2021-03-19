import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import json, os

from router import Router


class Trainer(ABC):
    
    def __init__(self, args):
        self.args = args
        self._router = Router(args, self.args.num_tasks).cuda()
        self.optimizer = optim.Adam(self._router.parameters(), lr=self.args.lr)
        self.adv_avg = torch.zeros(self.args.max_depth).cuda()
        self.adv_std = torch.ones(self.args.max_depth).cuda()
        self.critic_loss_fn = nn.SmoothL1Loss()
        self._layer_count = {}
        if self.args.load:
            self.load()

    @abstractmethod
    def train(self):
        pass


    def compute_return(self, rewards, action_probs, critic_outputs):
        returns = torch.zeros_like(rewards, dtype=torch.float32).cuda()
        adv = torch.zeros_like(rewards, dtype=torch.float32).cuda()
        T = len(returns)
        for t in reversed(range(T)):
            if t == T - 1:
                returns[t] = rewards[t]
                adv[t] = returns[t] -  critic_outputs[t].detach()
            else:
                returns[t] = rewards[t] + self.args.gamma * returns[t+1]

                # TD Residual
                delta = rewards[t] + self.args.gamma * critic_outputs[t+1].detach() -  critic_outputs[t].detach()
                
                # GAE
                adv[t] = delta + self.args.gamma * self.args.gae_lam * adv[t+1]
                #adv[t] = returns[t] - critic_outputs[t].detach()
        
        tdlamret = adv + torch.tensor(critic_outputs).cuda() 
        return returns, adv, tdlamret

    def layer_reward_bonus(self, actions, bonus_mask):
        bonus = torch.zeros(torch.sum(bonus_mask)).cuda()
        b_i = 0
        for i in range(len(bonus)):
            if bool(bonus_mask[i]):
                if actions[i] not in self._layer_count:
                    self._layer_count[actions[i]] = 0
                bonus[b_i] += (1 / np.log(self._layer_count[actions[i]] + 2))
                
                #bonus[b_i] += 1 / (self._layer_count[actions[i]] + 1)
                b_i += 1
                self._layer_count[actions[i]] += 1
        return bonus

    def save(self):
        print("Saving models")
        model_dict = {
            "router" : self._router.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "adv_avg" : self.adv_avg,
            "adv_std" : self.adv_std
        }
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        torch.save(model_dict, os.path.join(self.args.save_dir, "router.pt"))
        self._router.save_layers()

    def load(self):
        #self._router.load_state_dict(torch.load(os.path.join(self.args.save_dir, "router.pt")))
        model_dict = torch.load(os.path.join(self.args.save_dir, "router.pt"))
        self._router.load_state_dict(model_dict["router"])
        self.optimizer.load_state_dict(model_dict["optimizer"])
        self.adv_avg = model_dict["adv_avg"]
        self.adv_std = model_dict["adv_std"] 
        self._router.load_layers()

    def save_results(self, train_results):
        with open(self.args.results, "w") as f:
            json.dump(train_results, f)
