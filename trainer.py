import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import json, os

from router import Router
from fb_model import FBModel


class Trainer(ABC):
    
    def __init__(self, args):
        self.args = args
        self._fb_model = FBModel(self.args).cuda()
        self._router = Router(self.args, self.args.num_tasks, self._fb_model).cuda()
        
        self.optimizer = optim.Adam(self._router.parameters(), lr=self.args.lr)
        self.fb_optim = optim.Adam(self._fb_model.parameters(), lr=self.args.fb_lr)
        self.adv_avg = torch.zeros(self.args.max_depth).cuda()
        self.adv_std = torch.ones(self.args.max_depth).cuda()
        self.critic_loss_fn = nn.SmoothL1Loss()
        self._layer_count = {}
        if self.args.load:
            self.load()
        else:
            self.test_results = {
                "test_f1_scores" : [],
                "test_acc_scores" : [],
                "load_bal_loss" : [],
                "action_entropy": [],
                "num_unused_fbs" : []
            }


    @abstractmethod
    def train(self):
        pass


    def compute_return(self, rewards, action_probs, critic_outputs):
        returns = torch.zeros_like(rewards, dtype=torch.float32).cuda()
        adv = torch.zeros_like(rewards, dtype=torch.float32).cuda()
        T = len(returns[0])
        for t in reversed(range(T)):
            if t == T - 1:
                returns[:, t] = rewards[:, t]
                adv[:, t] = returns[:, t] -  critic_outputs[:, t, 0].detach()
            else:
                returns[:, t] = rewards[:, t] + self.args.gamma * returns[:, t+1]

                # TD Residual
                delta = rewards[:, t] + self.args.gamma * critic_outputs[:, t+1, 0].detach() -  critic_outputs[:, t, 0].detach()
                
                # GAE
                adv[:, t] = delta + self.args.gamma * self.args.gae_lam * adv[:, t+1]
                #adv[t] = returns[t] - critic_outputs[t].detach()
        
        tdlamret = adv + critic_outputs[:, :, 0].detach()
        return returns, adv, tdlamret

    def layer_reward_bonus(self, actions):
        batch_size = len(actions)
        depth = len(actions[0])
        bonus = torch.zeros((batch_size, depth)).cuda()
        for i in range(batch_size):
            for j in range(depth):
                if actions[i][j] not in self._layer_count:
                    self._layer_count[actions[i][j]] = 0
                bonus[i][j] += (1 / np.log(self._layer_count[actions[i][j]] + 2))
                    
                #bonus[i][j] += 1 / (self._layer_count[actions[i][j]] + 1)
                self._layer_count[actions[i][j]] += 1
        #print(bonus)
        return bonus

    def save(self):
        print("Saving models")
        model_dict = {
            "router" : self._router.state_dict(),
            "fb_model" : self._fb_model.state_dict(),
            "fb_optim" : self.fb_optim.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "adv_avg" : self.adv_avg,
            "adv_std" : self.adv_std
        }
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        torch.save(model_dict, os.path.join(self.args.save_dir, "router.pt"))
    
        with open(os.path.join(self.args.save_dir, self.args.test_results), "w") as f:
            json.dump(self.test_results, f)

    def load(self):
        #self._router.load_state_dict(torch.load(os.path.join(self.args.save_dir, "router.pt")))
        model_dict = torch.load(os.path.join(self.args.save_dir, "router.pt"))
        self._router.load_state_dict(model_dict["router"])
        self._fb_model.load_state_dict(model_dict["fb_model"])
        self.fb_optim.load_state_dict(model_dict["fb_optim"])

        self.optimizer.load_state_dict(model_dict["optimizer"])
        self.adv_avg = model_dict["adv_avg"]
        self.adv_std = model_dict["adv_std"]

        self.test_results = {}
        with open(os.path.join(self.args.save_dir, self.args.test_results), "r") as f:
            self.test_results = json.load(f)

