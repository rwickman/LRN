import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import json, os

from router import Router
from fb_model import FBModel
from metaworld_fb_model import MetaWorldFBModel
from model_util import PASS_FB

class Trainer(ABC):
    
    def __init__(self, args):
        self.args = args
        
        if self.args.metaworld:
            self._fb_model = MetaWorldFBModel(self.args).cuda()
        else:
            self._fb_model = FBModel(self.args).cuda()
        
        
        self._router = Router(self.args, self.args.num_tasks, self._fb_model.get_fb, self._fb_model.get_num_fbs()).cuda()
        print("ROUTER PARAMS: ", sum(p.numel() for p in self._router.parameters()))
        print("FB MODEL PARAMS: ", sum(p.numel() for p in self._fb_model.parameters()))
        
        self.optimizer = optim.Adam(self._router.parameters(), lr=self.args.lr)
        self.fb_optim = optim.Adam(self._fb_model.parameters(), lr=self.args.fb_lr)
        self.adv_avg = torch.zeros(self.args.max_depth).cuda()
        self.adv_std = torch.ones(self.args.max_depth).cuda()
        self.critic_loss_fn = nn.SmoothL1Loss()
        self._fb_count = torch.zeros(self._router.num_fbs + 1)
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
        with torch.no_grad():
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

            # Standardize advantages
            adv = (adv - adv.mean()) / adv.std()
            return returns, adv, tdlamret

    # def layer_reward_bonus(self, actions, output_losses_batch):
    #     batch_size = len(actions)
    #     depth = len(actions[0])
    #     bonus = torch.zeros((batch_size, depth)).cuda()
    #     temp_fb_count = self._fb_count.clone()
    #     temp_fb_count[PASS_FB] = 0
    #     fb_count_max = temp_fb_count.max()
    #     fb_count_min = temp_fb_count.min()
        
    #     for i in range(batch_size):
    #         for j in range(depth):
    #             total_reward[actions[i][j]] += output_losses_batch[i]
    #     #avg_reward = avg_reward / (temp_fb_count += 1)
    #     for i in range(batch_size):
    #         for j in range(depth):
    #             if actions[i][j] != PASS_FB:
    #                 bonus[i][j] += 1 - (temp_fb_count[actions[i][j]] - fb_count_min) / (fb_count_max - fb_count_min)
                
    #             #bonus[i][j] = avg_reward[actions[i][j]]
    #             #bonus[i][j] += 1 - self._fb_count[actions[i][j]]  / self._fb_count.sum()   
    #     #bonus = -(avg_reward.max()  - avg_reward / avg_reward.sum())

    #     return bonus


    def layer_reward_bonus(self, actions, output_losses_batch):
        batch_size = len(actions)
        depth = len(actions[0])
        bonus = torch.zeros((batch_size, depth)).cuda()
        temp_fb_count = self._fb_count.clone()
        temp_fb_count[PASS_FB] = 0
        fb_count_max = temp_fb_count.max()
        fb_count_min = temp_fb_count.min()
        
        # Will need to calculate two things:
        # 1. (num times used / # eps) for each action
            # a. ANS: fb_count / batch_Size
        # 2. P(not used at all)
            # a. How to know if a layer 

        for i in range(batch_size):
            for j in range(depth):
                if actions[i][j] != PASS_FB:
                    fb_type = self._fb_model.action_to_fb_type(actions[i][j])
                    # Get the probability that equals the uniform sampling of this FB 
                    uni_fb_prob = self._fb_model.get_fb_prob(fb_type)
                    cur_fb_prob = self._fb_count[actions[i][j]] / batch_size
                    log_loss = -(uni_fb_prob * torch.log(cur_fb_prob + 1e-5) + (1-uni_fb_prob) * torch.log((1-cur_fb_prob) + 1e-5) )
                    prob_diff = (uni_fb_prob - cur_fb_prob)
                    if abs(prob_diff) <= 0.05:
                        bonus[i][j] += 1
                    elif prob_diff < 0:
                        bonus[i][j] += -log_loss
                    else:
                        bonus[i][j] += 1 - log_loss


                    #log_loss = -(uni_fb_prob * torch.log(cur_fb_prob + 1e-5) + (1-uni_fb_prob) * torch.log((1-cur_fb_prob) + 1e-5) )
                    #bonus[i][j] += -log_loss#bonus[i][j]
                else:
                    bonus[i][j] += -5
                    # prob_diff = (uni_fb_prob - cur_fb_prob)**
                    # if abs(prob_diff) < 0.1:
                    #     bonus[i][j] += 5
                    # else:     
                    #     bonus[i][j] += -5
                    
                #         if cur_fb_prob < uni_fb_prob:
                        
                # else:
                #     bonus[i][j] += -5
                    # if cur_fb_prob <= uni_fb_prob:
                    #     bonus[i][j] += (1 - (uni_fb_prob - cur_fb_prob)) * 5
                    # else:
                    #      bonus[i][j] += (uni_fb_prob - cur_fb_prob) * 5
                #bonus[i][j] = avg_reward[actions[i][j]]
                #bonus[i][j] += 1 - self._fb_count[actions[i][j]]  / self._fb_count.sum()   
        #bonus = -(avg_reward.max()  - avg_reward / avg_reward.sum())
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
        
        with open(os.path.join(self.args.save_dir, "fb_count.json"), "w") as f:
            json.dump(self._fb_count.tolist(), f)

        with open(os.path.join(self.args.save_dir,"args.json"), "w") as f:
            json.dump(vars(self.args), f)

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
        
        fb_count_json = os.path.join(self.args.save_dir, "fb_count.json")
        if os.path.exists(fb_count_json):
            with open(fb_count_json) as f:
                self._fb_count = torch.tensor(json.load(f))


