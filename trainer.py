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
from model_util import window_and_pad

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
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda e: self.args.lr_gamma)
        self.fb_optim = optim.Adam(self._fb_model.parameters(), lr=self.args.fb_lr)
        self._fb_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.fb_optim, lr_lambda=lambda e: self.args.fb_lr_gamma)
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


    def compute_return(self, rewards, critic_outputs):
        with torch.no_grad():
            returns = torch.zeros_like(rewards, dtype=torch.float32).cuda()
            advs = torch.zeros_like(rewards, dtype=torch.float32).cuda()
            T = len(returns[0])
            for t in reversed(range(T)):
                if t == T - 1:
                    returns[:, t] = rewards[:, t]
                    advs[:, t] = returns[:, t] -  critic_outputs[:, t, 0].detach()
                else:
                    returns[:, t] = rewards[:, t] + self.args.gamma * returns[:, t+1]

                    # TD Residual
                    delta = rewards[:, t] + self.args.gamma * critic_outputs[:, t+1, 0].detach() -  critic_outputs[:, t, 0].detach()
                    
                    # GAE
                    advs[:, t] = delta + self.args.gamma * self.args.gae_lam * advs[:, t+1]
                    #advs[t] = returns[t] - critic_outputs[t].detach()
            
            tdlamret = advs + critic_outputs[:, :, 0].detach()

            # Standardize advantages
            advs = (advs - advs.mean()) / advs.std()
            return returns, advs, tdlamret


    def compute_load_balance_loss(self, valid_fbs_count, valid_fbs_ep_pad, new_prob, actions_batch):
        kl = torch.nn.KLDivLoss()
        action_prob_sum = torch.zeros(self._router.num_fbs + 1, 1).cuda()        
        batch_size = valid_fbs_ep_pad.shape[0]
        action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()
        #total_smooth_loss = torch.zeros(1).cuda()
        
        for b_i in range(batch_size):
            num_valid_fbs = valid_fbs_count[b_i].int()
            action_prob_sum[valid_fbs_ep_pad[b_i][:num_valid_fbs]] += new_prob[b_i][:num_valid_fbs].unsqueeze(1)
            action_count[valid_fbs_ep_pad[b_i][:num_valid_fbs]] += 1

        # Get probability of each FB over batch
        action_idxs = action_count.squeeze(-1) > 0
        action_prob_sum[action_idxs] = action_prob_sum[action_idxs] / action_count[action_idxs]
        #print("action_prob_prob", action_prob_sum, "action_prob_sum.sum()", action_prob_sum.sum())
        #print("self._fb_model.target_prob", self._fb_model.target_prob, "self._fb_model.target_prob.sum()", self._fb_model.target_prob.sum())
        # Cross-entropy loss
        load_balance_loss = torch.mean(-self._fb_model.target_prob * torch.log(action_prob_sum + 1e-9))
        #print("load_balance_loss", load_balance_loss)
        return load_balance_loss

    # def layer_reward_bonus(self, actions, output_losses_batch):
    #     import time
    #     start_layer_time  = time.time()
    #     batch_size = len(actions)
    #     depth = len(actions[0])
    #     bonus = torch.zeros((batch_size, depth)).cuda()
    #     temp_fb_count = self._fb_count.clone()
    #     temp_fb_count[PASS_FB] = 0
    #     fb_count_max = temp_fb_count.max()
    #     fb_count_min = temp_fb_count.min()
        
    #     # 1. Get counts for type
    #     # 2. If not all equal:
    #         # a. Give negative reward to one with most count
    #         # b. Give positive reward to one with least count
    #         # (Consider in future make this some sort of gradient)
    #     # Iterate over actions, and assign the reward based on the fb_bonus
    #     fb_bonus = torch.zeros(self._router.num_fbs)
    #     fb_type_counts = torch.zeros(self._fb_model.num_fb_types, self.args.num_dup_layers)
    #     for a_i in range(1, self._router.num_fbs+1):
    #         fb_type = (a_i-1) % self._fb_model.num_fb_types
    #         dup_num = (a_i-1)//self._fb_model.num_fb_types
    #         #print(fb_type, dup_num, a_i)
    #         fb_type_counts[fb_type, dup_num] = self._fb_count[a_i]
        
    #     #print("fb_type_counts", fb_type_counts)
        
    #     fb_maxes = fb_type_counts.argmax(axis=1)
        
    #     fb_type_sums = fb_type_counts.sum() #fb_type_counts.sum(axis=1).unsqueeze(1)
    #     # print("fb_maxes", fb_maxes)
    #     # print("fb_type_sums", fb_type_sums)
    #     # print("fb_type_counts", fb_type_counts)
    #     fb_type_avgs = fb_type_counts / fb_type_sums

    #     #print("fb_type_avgs", fb_type_avgs)
    #         # self._fb_count[a_i]
        
    #     for i in range(batch_size):
    #         for j in range(depth):
    #             if actions[i][j] == 0:
    #                 bonus[i][j] = -1
    #             fb_type = (actions[i][j]-1) % self._fb_model.num_fb_types
    #             dup_num = (actions[i][j]-1)//self._fb_model.num_fb_types
    #             #print("fb_type", fb_type, "fb_maxes[fb_type]", fb_maxes[fb_type], "dup_num", dup_num)
    #             if fb_type_avgs[fb_type, dup_num] > 1/self.args.num_dup_layers:
    #                 bonus[i][j] = -1#actions[i][j]
    #             else:
    #                 bonus[i][j] = 1
        
    #     return bonus


    def layer_reward_bonus(self, actions, output_losses_batch):
        import time
        start_layer_time  = time.time()
        batch_size = len(actions)
        depth = len(actions[0])
        bonus = torch.zeros((batch_size, depth)).cuda()

        fb_avgs = self._fb_count / self._fb_count.sum()
        fb_count_min = self._fb_count.min()
        fb_count_max = self._fb_count.max()
        # print("fb_avgs", fb_avgs)
        # print("fb_avgs.sum()", fb_avgs.sum())
        #print("fb_type_avgs", fb_type_avgs)
            # self._fb_count[a_i]
        self._fb_count[PASS_FB] = 0
        for i in range(batch_size):
            for j in range(depth):
                if actions[i][j] == PASS_FB:
                    bonus[i][j] = 0
                else:
                    #print("fb_type", fb_type, "fb_maxes[fb_type]", fb_maxes[fb_type], "dup_num", dup_num)
                    bonus[i][j] += 1 - (self._fb_count[actions[i][j]] - fb_count_min) / (fb_count_max - fb_count_min)
                # if fb_avgs[actions[i][j]] > 1/self.args.num_dup_layers:
                #     bonus[i][j] = -1
                # else:
                #     bonus[i][j] = 1
        print("BONUS TIME", time.time() - start_layer_time)
        return bonus


    # def layer_reward_bonus(self, actions, output_losses_batch):
    #     batch_size = len(actions)
    #     depth = len(actions[0])
    #     bonus = torch.zeros((batch_size, depth)).cuda()
    #     temp_fb_count = self._fb_count.clone()
    #     temp_fb_count[PASS_FB] = 0
    #     fb_count_max = temp_fb_count.max()
    #     fb_count_min = temp_fb_count.min()
        
    #     # Will need to calculate two things:
    #     # 1. (num times used / # eps) for each action
    #         # a. ANS: fb_count / batch_Size
    #     # 2. P(not used at all)
    #         # a. How to know if a layer 

    #     for i in range(batch_size):
    #         for j in range(depth):
    #             if actions[i][j] != PASS_FB:
    #                 fb_type = self._fb_model.action_to_fb_type(actions[i][j])
    #                 # Get the probability that equals the uniform sampling of this FB 
    #                 uni_fb_prob = self._fb_model.get_fb_prob(fb_type)
    #                 cur_fb_prob = self._fb_count[actions[i][j]] / batch_size
    #                 log_loss = -(uni_fb_prob * torch.log(cur_fb_prob + 1e-5) + (1-uni_fb_prob) * torch.log((1-cur_fb_prob) + 1e-5) )
    #                 prob_diff = (uni_fb_prob - cur_fb_prob)
    #                 if abs(prob_diff) <= 0.05:
    #                     bonus[i][j] += 1
    #                 elif prob_diff < 0:
    #                     bonus[i][j] += -log_loss
    #                 else:
    #                     bonus[i][j] += 1 - log_loss


    #                 #log_loss = -(uni_fb_prob * torch.log(cur_fb_prob + 1e-5) + (1-uni_fb_prob) * torch.log((1-cur_fb_prob) + 1e-5) )
    #                 #bonus[i][j] += -log_loss#bonus[i][j]
    #             else:
    #                 bonus[i][j] += -5
    #                 # prob_diff = (uni_fb_prob - cur_fb_prob)**
    #                 # if abs(prob_diff) < 0.1:
    #                 #     bonus[i][j] += 5
    #                 # else:     
    #                 #     bonus[i][j] += -5
                    
    #             #         if cur_fb_prob < uni_fb_prob:
                        
    #             # else:
    #             #     bonus[i][j] += -5
    #                 # if cur_fb_prob <= uni_fb_prob:
    #                 #     bonus[i][j] += (1 - (uni_fb_prob - cur_fb_prob)) * 5
    #                 # else:
    #                 #      bonus[i][j] += (uni_fb_prob - cur_fb_prob) * 5
    #             #bonus[i][j] = avg_reward[actions[i][j]]
    #             #bonus[i][j] += 1 - self._fb_count[actions[i][j]]  / self._fb_count.sum()   
    #     #bonus = -(avg_reward.max()  - avg_reward / avg_reward.sum())
    #     return bonus

    def train_router(self, valid_fbs_ep, actions, action_indices, action_probs, max_fbs, x_batch_steps, task_ids_steps, returns, advs, tdlamret, critic_outputs):
        critic_losses = []
        actor_losses = []
        load_balance_losses = []

        actions_mask, actions_window, valid_fbs_ep_pad, actions_one_hot, action_probs_pad, valid_fbs_count, actions_batch = window_and_pad(
            valid_fbs_ep,
            actions,
            action_indices,
            action_probs,
            self.args.max_depth,
            max_fbs)

        x_batch_steps = torch.repeat_interleave(x_batch_steps, (self.args.max_depth), axis=0)
        task_ids_steps = torch.repeat_interleave(task_ids_steps, self.args.max_depth)
        num_exs = len(valid_fbs_ep_pad)

        with torch.autograd.set_detect_anomaly(True):
            total_critic_loss = 0
            total_actor_loss = 0
            total_load_balance_loss = 0
            for e_i in range(self.args.noptepochs):
                batch_ids = torch.randperm(num_exs)

                for m_i in range(self.args.nminibatches):
                    loss = torch.zeros(1).cuda()
                    start = m_i * (num_exs // self.args.nminibatches)
                    end = (m_i + 1) * (num_exs // self.args.nminibatches)
                    m_batch_idxs = batch_ids[start:end]
                    # Get prediction for current timestep
                    new_prob, val_pred = self._router.predict_step(
                        x_batch_steps[m_batch_idxs],
                        task_ids_steps[m_batch_idxs],
                        valid_fbs_ep_pad[m_batch_idxs],
                        actions_window[m_batch_idxs],
                        actions_mask[m_batch_idxs])
                    
                    actor_loss, critic_loss = self.ppo_loss(
                        advs[m_batch_idxs], 
                        action_probs_pad[m_batch_idxs], 
                        new_prob, 
                        tdlamret[m_batch_idxs], 
                        val_pred,
                        actions_one_hot[m_batch_idxs],
                        critic_outputs[m_batch_idxs].unsqueeze(-1))

                    

                    loss += actor_loss + self.args.critic_lam * critic_loss

                    total_critic_loss += self.args.critic_lam * critic_loss.detach().cpu().numpy()
                    total_actor_loss += actor_loss.detach().cpu().numpy()
                    if self.args.load_bal_lam > 0:
                        load_balance_loss = self.args.load_bal_lam * self.compute_load_balance_loss(valid_fbs_count[m_batch_idxs], valid_fbs_ep_pad[m_batch_idxs], new_prob, actions_batch[m_batch_idxs]) 
                        total_load_balance_loss += load_balance_loss.detach().cpu().numpy()
                        loss += load_balance_loss                          

                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(self._router.parameters(), 2.0)
                    print("self._router.critic.critic_out1.weight.grad.max()", self._router.critic.critic_out1.weight.grad.max())
                    print("self._router.critic.critic_out2.weight.grad.max()", self._router.critic.critic_out2.weight.grad.max())
                    print("self._router.state_encoder.fc1.weight.grad.max()", self._router.state_encoder.fc1.weight.grad.max())
                    print("self._router.state_encoder.fc3.weight.grad.max()", self._router.state_encoder.fc3.weight.grad.max())
                    print("self._router.state_encoder.fc3.weight.grad.min()", self._router.state_encoder.fc3.weight.grad.min())
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                critic_losses.append(total_critic_loss)
                actor_losses.append(total_actor_loss)
                load_balance_losses.append(total_load_balance_loss)

        
        print("Critic Outputs", val_pred[:5].squeeze(-1).detach().cpu().numpy(), "VS", tdlamret[m_batch_idxs][:5].detach().cpu().numpy())
        print("ADVS", advs[m_batch_idxs][:5].detach().cpu().numpy())
        print("critic_losses", critic_losses)
        return actor_losses, critic_losses, load_balance_losses

    def save(self):
        print("Saving models")
        model_dict = {
            "router" : self._router.state_dict(),
            "fb_model" : self._fb_model.state_dict(),
            "fb_optim" : self.fb_optim.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "router_lr_scheduler" : self._lr_scheduler.state_dict(),
            "fb_lr_scheduler" : self._fb_lr_scheduler.state_dict(),
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
        self._lr_scheduler.load_state_dict(model_dict["router_lr_scheduler"])
        self._fb_lr_scheduler.load_state_dict(model_dict["fb_lr_scheduler"])

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


    def ppo_loss(self, adv, old_prob, new_prob, val_true, val_pred, actions_one_hot, old_val_pred):
        """PPO discreate loss function.
        Args:
            advantage: advantage from the episode.
            old_pred: old softmax predictions.
            y_true: one-hot encoded actions taken in the episode.
            y_pred: new softmax predictions.
            val_true: true values (i.e., returns) from the episode.
            val_pred: critic value predictions
        
        Returns:
            A tuple of PPO loss and value MSE loss 
        """
        new_prob = torch.log(torch.sum(actions_one_hot * new_prob, axis=-1))
        old_prob = torch.log(torch.sum(actions_one_hot * old_prob, axis=-1))

        # Compute ratio (NOTE: this is same as ratio = new_prob/old_prob) 
        ratio = torch.exp(new_prob - old_prob.detach())
        surrogate_1 = ratio * adv

        # Clipped policy gradient, ratio clipped between [0.8, 1.2]
        surrogate_2 = torch.clip(ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * adv

        # Entropy regularization
        #entropy_pen = -ENTROPY_COEF * tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=1)
        entropy_pen = 0
        # PPO loss, averaged over several batches
        policy_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))

        # Value loss (MSE)
        val_true = val_true.unsqueeze(-1).unsqueeze(-1)

        # old_val_pred =old_val_pred.unsqueeze(-1)

        # val_pred_clipped = old_val_pred + torch.clip(val_pred - old_val_pred, -self.args.val_clip, self.args.val_clip) 

        # # Unclipped value loss
        # vf_losses1 = torch.square(val_pred - val_true)

        # # Clipped value loss
        # vf_losses2 = torch.square(val_pred_clipped - val_true)
        # vf_loss = torch.mean(torch.max(vf_losses1, vf_losses2))

        vf_loss = self.critic_loss_fn(val_pred, val_true)
        #print("policy_loss", policy_loss, "vf_loss", vf_loss)
        return  policy_loss, vf_loss