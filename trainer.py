import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import json, os
from torch.distributions import Categorical

from router import Router
from model_util import PASS_FB, create_lookahead_mask

class Trainer(ABC):
    
    def __init__(self, args, mb_model):
        self.args = args
        self._mb_model = mb_model.cuda()
        self.debug = False
        self._router = Router(
            self.args,
            self.args.num_tasks,
            self._mb_model.get_mb,
            self._mb_model.get_num_mbs()).cuda()
        print("ROUTER PARAMS: ", sum(p.numel() for p in self._router.parameters()))
        print("FB MODEL PARAMS: ", sum(p.numel() for p in self._mb_model.parameters()))
        
        self.optimizer = optim.Adam(self._router.parameters(), lr=self.args.lr, eps=1e-4)
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lr_lambda=lambda e: self.args.lr_gamma)
        self.mb_optim = optim.Adam(self._mb_model.parameters(), lr=self.args.mb_lr, eps=1e-4)
        self._mb_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(self.mb_optim, lr_lambda=lambda e: self.args.mb_lr_gamma)
        self.adv_avg = torch.zeros(self.args.max_depth).cuda()
        self.adv_std = torch.ones(self.args.max_depth).cuda()
        self.critic_loss_fn = nn.SmoothL1Loss()
        self._action_smooth_loss = nn.KLDivLoss()
        self._mb_count = torch.zeros(self._router.num_mbs + 1)
        if self.args.load:
            self.load()
        else:
            self.test_results = {
                "test_f1_scores" : [],
                "test_acc_scores" : [],
                "load_bal_loss" : [],
                "action_entropy": [],
                "num_unused_mbs" : []
            }

        # for param_group in self.mb_optim.param_groups:
        #     print("mb_optim lr", param_group["lr"])
        #     param_group["lr"] = 1e-4
        # for param_group in self.optimizer.param_groups:
        #     #print("mb_optim lr", param_group["lr"])
        #     param_group["lr"] = 1e-4
        
    @abstractmethod
    def train(self):
        pass


    def compute_return(self, rewards, critic_outputs, dones=None):
        with torch.no_grad():
            returns = torch.zeros_like(rewards, dtype=torch.float32).cuda()
            advs = torch.zeros_like(rewards, dtype=torch.float32).cuda()
            T = returns.shape[1]
            for t in reversed(range(T)):
                if t == T - 1:
                    cur_terminal = 1.0 - dones[:, t] if dones is not None else 0.0
                    returns[:, t] = (1 - cur_terminal) * rewards[:, t] + cur_terminal * critic_outputs[:, t]
                    advs[:, t] = returns[:, t] -  critic_outputs[:, t].detach()
                else:
                    # Determines if the current state is a terminal state
                    cur_terminal = 1.0 - dones[:, t] if dones is not None else 1.0
                    
                    # Discounted return
                    returns[:, t] = rewards[:, t] + self.args.gamma * returns[:, t+1] * cur_terminal

                    # TD Residual
                    delta = rewards[:, t] + self.args.gamma * critic_outputs[:, t+1].detach() * cur_terminal -  critic_outputs[:, t].detach() 
                    
                    # GAE
                    advs[:, t] = delta + self.args.gamma * self.args.gae_lam * advs[:, t+1] * cur_terminal
                    #advs[:, t] = returns[:. t] - critic_outputs[:, t].detach()
            
            tdlamret = advs + critic_outputs[:, :].detach()

            # Standardize advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)
            return returns, advs, tdlamret


    def compute_load_balance_loss(self, valid_mbs_count, valid_mbs_ep_pad, new_prob, actions_batch):
        kl = torch.nn.KLDivLoss()
        action_prob_sum = torch.zeros(self._router.num_mbs + 1, 1).cuda()        
        batch_size = valid_mbs_ep_pad.shape[0]
        action_count = torch.zeros(self._router.num_mbs + 1, 1).cuda()
        #total_smooth_loss = torch.zeros(1).cuda()
        
        for b_i in range(batch_size):
            num_valid_mbs = valid_mbs_count[b_i].int()
            action_prob_sum[valid_mbs_ep_pad[b_i][:num_valid_mbs]] += new_prob[b_i][:num_valid_mbs].unsqueeze(1)
            action_count[valid_mbs_ep_pad[b_i][:num_valid_mbs]] += 1

        # Get probability of each FB over batch
        action_idxs = action_count.squeeze(-1) > 0
        action_prob_sum[action_idxs] = action_prob_sum[action_idxs] / action_count[action_idxs]
        #print("action_prob_prob", action_prob_sum, "action_prob_sum.sum()", action_prob_sum.sum())
        #print("self._mb_model.target_prob", self._mb_model.target_prob, "self._mb_model.target_prob.sum()", self._mb_model.target_prob.sum())
        # Cross-entropy loss
        load_balance_loss = torch.mean(-self._mb_model.target_prob * torch.log(action_prob_sum + 1e-9))
        #print("load_balance_loss", load_balance_loss)
        return load_balance_loss


    def layer_reward_bonus(self, actions):
        # import time
        # start_layer_time  = time.time()
        batch_size = len(actions)
        depth = len(actions[0])
        bonus = torch.zeros_like(actions).cuda()
        #mb_avgs = self._mb_count / self._mb_count.sum()
        mb_count_min = self._mb_count.min()
        mb_count_max = self._mb_count.max()
        # print("mb_avgs", mb_avgs)
        # print("mb_avgs.sum()", mb_avgs.sum())
        #print("mb_type_avgs", mb_type_avgs)
            # self._mb_count[a_i]

        bonus_factor = 1 - (len(self._mb_count) - self._mb_count.count_nonzero())/len(self._mb_count)
        #print("self._mb_count", self._mb_count)
        #print("bonus_factor", bonus_factor)
        if not self.args.rl:
            self._mb_count[PASS_FB] = 0
        # print("bonus_factor", bonus_factor)
        bonus = bonus_factor  - (self._mb_count[actions].cuda() - mb_count_min) / (mb_count_max - mb_count_min + 1e-6).cuda()
        bonus = bonus * 0.1

        # for i in range(batch_size):
        #     for j in range(depth):
        #         if not self.args.rl and actions[i][j] == PASS_FB:
        #             bonus[i][j] = 0
        #         else:
        #             #print("mb_type", mb_type, "mb_maxes[mb_type]", mb_maxes[mb_type], "dup_num", dup_num)
        #             bonus[i][j] += 1 - (self._mb_count[actions[i][j]] - mb_count_min) / (mb_count_max - mb_count_min + 1e-6)
                # if mb_avgs[actions[i][j]] > 1/self.args.num_dup_layers:
                #     bonus[i][j] = -1
                # else:
                #     bonus[i][j] = 1
        #print("BONUS TIME", time.time() - start_layer_time)
        
        #print("bonus", bonus)
        return bonus


    def train_router(self, imgs, task_ids, rewards, router_results):
        """Train the router on batches of experience."""
        critic_losses = []
        actor_losses = []
        batch_size = len(imgs)
        self._mb_count = router_results.mb_count
        if not self.args.no_reward_bonus:
            bonus = self.layer_reward_bonus(router_results.actions)
            rewards += bonus
        

        returns, advs, tdlamret = self.compute_return(rewards, router_results.critic_outputs)
        advs = advs.reshape(-1)
        tdlamret = tdlamret.reshape(-1)
        returns = returns.reshape(-1)

        # TODO: Update this to change shape and need to change predict_step critic reshaping as well
        num_patches = (self.args.img_size**2) // self.args.patch_size**2
        
        if self.args.rl:
            # pred emb + task_emb + state emb + max_depth
            #max_inp = 3 + num_patches
            max_inp = 3
        else:
            # class emb + task_emb + num_patches + max_depth
            max_inp = 2 + num_patches 

        # Create lookhead masks for module predictions to mask out future timesteps
        mb_masks = create_lookahead_mask(self.args.max_depth, batch_size, max_inp).cuda()

        # Reshape so you can treat each timestep as an example
        actions_final = router_results.actions.reshape(-1)
        action_masks = router_results.action_masks.view(-1, router_results.action_masks.shape[2])

        # Create actions per timestep, and remove the last timestep actions as input
        actions_inp = torch.repeat_interleave(router_results.actions, (self.args.max_depth), axis=0)

        # Repeat images and tasks IDs for each timstep
        imgs_batch_steps = torch.repeat_interleave(imgs, (self.args.max_depth), axis=0)
        task_ids_steps = torch.repeat_interleave(task_ids, (self.args.max_depth), axis=0)

        num_exs = task_ids_steps.shape[0]


        router_results.action_probs = router_results.action_probs.reshape(-1, router_results.action_probs.shape[-1])
        router_results.critic_outputs = router_results.critic_outputs.reshape(-1)

        with torch.autograd.set_detect_anomaly(True):

            # TODO: Use a DataLoader for this
            #print("num_exs", num_exs)
            #print("batch_size", batch_size)
            for e_i in range(self.args.noptepochs):

                batch_ids = torch.randperm(num_exs)
                for m_i in range(self.args.nminibatches):
                    start = m_i * (num_exs // self.args.nminibatches)
                    end = (m_i + 1) * (num_exs // self.args.nminibatches)
                    m_batch_idxs = batch_ids[start:end]
             
                    # Get prediction for current timestep
                    
                    new_prob, val_pred = self._router.predict_step(
                        imgs_batch_steps[m_batch_idxs],
                        task_ids_steps[m_batch_idxs],
                        actions_inp[m_batch_idxs],
                        mb_masks[m_batch_idxs],
                        action_masks[m_batch_idxs])
                    
                    actor_loss, critic_loss, should_stop = self.ppo_loss(
                        advs[m_batch_idxs], 
                        router_results.action_probs[m_batch_idxs], 
                        new_prob, 
                        returns[m_batch_idxs], 
                        val_pred.view(-1),
                        actions_final[m_batch_idxs].unsqueeze(1),
                        router_results.critic_outputs[m_batch_idxs])

                    if should_stop:
                        print("\nnew_prob[:5]", new_prob[:5])
                        print("task_ids_steps[m_batch_idxs][:5]", task_ids_steps[m_batch_idxs][:5])
                        #print("Critic Outputs", val_pred[:5].squeeze(-1).detach().cpu().numpy(), "VS", returns[m_batch_idxs][:5].detach().cpu().numpy())
                        print("ADVS", advs[m_batch_idxs][:5].detach().cpu().numpy())
                        print("Critic Outputs", val_pred[:50].squeeze(-1).detach().cpu().numpy(), "VS", returns[m_batch_idxs][:50].detach().cpu().numpy())
                        print("critic_losses", critic_losses)

                        print("self._mb_count", self._mb_count)
                        return actor_losses, critic_losses
                    
                    loss = actor_loss + self.args.critic_lam * critic_loss                        

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._router.parameters(), self.args.max_grad_norm)
                
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                    critic_losses.append(self.args.critic_lam * critic_loss.detach().cpu().numpy())
                    actor_losses.append(actor_loss.detach().cpu().numpy())

        if self.debug:
            print("new_prob[:5]", new_prob[:5])
            print("task_ids_steps[m_batch_idxs][:5]", task_ids_steps[m_batch_idxs][:5])
            #print("Critic Outputs", val_pred[:5].squeeze(-1).detach().cpu().numpy(), "VS", tdlamret[m_batch_idxs][:5].detach().cpu().numpy())
            print("ADVS", advs[m_batch_idxs][:5].detach().cpu().numpy())
            print("Critic Outputs", val_pred[:50].squeeze(-1).detach().cpu().numpy(), "VS", returns[m_batch_idxs][:50].detach().cpu().numpy())
            print("critic_losses", critic_losses)
            print("self._mb_count", self._mb_count)
    
        return actor_losses, critic_losses

    def save(self):
        print("Saving models")
        model_dict = {
            "router" : self._router.state_dict(),
            "mb_model" : self._mb_model.state_dict(),
            "mb_optim" : self.mb_optim.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "router_lr_scheduler" : self._lr_scheduler.state_dict(),
            "mb_lr_scheduler" : self._mb_lr_scheduler.state_dict(),
            "adv_avg" : self.adv_avg,
            "adv_std" : self.adv_std
        }
        
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        torch.save(model_dict, os.path.join(self.args.save_dir, "router.pt"))
    
        with open(os.path.join(self.args.save_dir, self.args.test_results), "w") as f:
            json.dump(self.test_results, f)
        
        with open(os.path.join(self.args.save_dir, "mb_count.json"), "w") as f:
            json.dump(self._mb_count.tolist(), f)

        with open(os.path.join(self.args.save_dir,"args.json"), "w") as f:
            json.dump(vars(self.args), f)

    def load(self):
        #self._router.load_state_dict(torch.load(os.path.join(self.args.save_dir, "router.pt")))
        model_dict = torch.load(os.path.join(self.args.save_dir, "router.pt"))
        self._router.load_state_dict(model_dict["router"])
        self._mb_model.load_state_dict(model_dict["mb_model"])
        self.mb_optim.load_state_dict(model_dict["mb_optim"])
        self._lr_scheduler.load_state_dict(model_dict["router_lr_scheduler"])
        self._mb_lr_scheduler.load_state_dict(model_dict["mb_lr_scheduler"])

        self.optimizer.load_state_dict(model_dict["optimizer"])
        self.adv_avg = model_dict["adv_avg"]
        self.adv_std = model_dict["adv_std"]



        self.test_results = {}
        with open(os.path.join(self.args.save_dir, self.args.test_results), "r") as f:
            self.test_results = json.load(f)
        
        mb_count_json = os.path.join(self.args.save_dir, "mb_count.json")
        if os.path.exists(mb_count_json):
            with open(mb_count_json) as f:
                self._mb_count = torch.tensor(json.load(f))


    def ppo_loss(self, adv, old_prob, new_prob, val_true, val_pred, actions, old_val_pred, return_step_loss=False):
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
        
        new_distr = Categorical(new_prob)
        old_distr = Categorical(old_prob)

        
        new_prob = new_distr.log_prob(actions.squeeze())

        old_prob = old_distr.log_prob(actions.squeeze())

        
        ratio = torch.exp(new_prob - old_prob)
        surrogate_1 = ratio * adv
        # Clipped policy gradient, ratio clipped between [0.8, 1.2]
        surrogate_2 = torch.clip(ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * adv

        # Entropy regularization
        entropy_pen = -self.args.entropy_lam * new_distr.entropy()
        #print("Entropy pen", entropy_pen.mean())
        # PPO loss, averaged over several batches
        policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
        #print("policy_loss", policy_loss)
        policy_loss = policy_loss + entropy_pen.mean()
        # policy_loss = -(torch.log(new_prob) * adv).mean()

        # Value loss (MSE)
        #print("val_true", val_true[:5])
        #print("val_pred", val_pred.squeeze(-1).squeeze(-1)[:5], val_pred.shape)
        # val_true = val_true.unsqueeze(-1).unsqueeze(-1)

        #old_val_pred =old_val_pred.unsqueeze(-1)
        #val_pred_clipped = old_val_pred + torch.clip(val_pred - old_val_pred, -self.args.val_clip, self.args.val_clip) 

        # Unclipped value loss
        #vf_losses1 = torch.square(val_true - val_pred)

        # Clipped value loss
        #vf_losses2 = torch.square(val_true - val_pred_clipped)
        #vf_loss = torch.mean(torch.max(vf_losses1, vf_losses2))
        
        vf_loss = self.critic_loss_fn(val_pred, val_true)

        # Get apporximate KL between new and old policies
        if return_step_loss:
            approx_kl = (old_prob.detach() - new_prob.detach()).mean().item()
        else:
            # Only get nonzero idxs for training router 
            nonzero_idxs = old_prob.flatten().nonzero()
            approx_kl = (old_prob.detach().flatten()[nonzero_idxs] - new_prob.detach().flatten()[nonzero_idxs]).mean().item()

        if self.debug:
            print("approx_kl", approx_kl)
        #print("1.5 * self.args.tgt_kl", 1.5 * self.args.tgt_kl)
        should_stop = approx_kl > self.args.tgt_kl



        if return_step_loss:
            critic_step_loss = (val_true - val_pred).detach() ** 2
            return policy_loss, vf_loss, critic_step_loss, should_stop

        #print("val_pred", val_pred, "val_true", val_true)
        
        
        #vf_loss = self.critic_loss_fn(val_pred.view(val_pred.shape[0]), val_true)
        #print("policy_loss", policy_loss, "vf_loss", vf_loss)
        return  policy_loss, vf_loss, should_stop


    def label_smooth_loss(self, action_indices, new_prob, valid_mbs_count, actions_one_hot, m_batch_idxs):
        """"Add label-smoothing loss."""
        batch_size = new_prob.shape[0]
        total_smooth_loss = torch.zeros(1).cuda()
        action_probs = torch.sum(actions_one_hot * new_prob, axis=-1)
        
        b_i_old = m_batch_idxs // self.args.max_depth
        t_old = m_batch_idxs % self.args.max_depth
        for b_i in range(batch_size):
            num_valid_mbs = valid_mbs_count[b_i].int()
            if num_valid_mbs > 1:
                max_action_prob = (1-self.args.alpha) + self.args.alpha/num_valid_mbs
                if action_probs[b_i] > max_action_prob:
                    smooth_actions = torch.zeros_like(new_prob[b_i])
                    # Add prob to all valid actions
                    smooth_actions[:num_valid_mbs] = self.args.alpha/num_valid_mbs

                    
                    # Add extra prob to selected action
                    smooth_actions[action_indices[b_i_old[b_i]][t_old[b_i]]] = max_action_prob
                    smooth_loss = self.args.smooth_lam * self._action_smooth_loss(torch.log(new_prob[b_i] + 1e-7), smooth_actions)
                    total_smooth_loss += smooth_loss
        print("total_smooth_loss", total_smooth_loss)
        return total_smooth_loss / batch_size

    def decay_lr(self):
        if self.optimizer.param_groups[0]["lr"] > self.args.min_lr:
            self._lr_scheduler.step()
        else:
            self.optimizer.param_groups[0]["lr"] = self.args.min_lr
        
        if self.mb_optim.param_groups[0]["lr"] > self.args.mb_min_lr:
            self._mb_lr_scheduler.step()
        else:
            self.mb_optim.param_groups[0]["lr"] = self.args.mb_min_lr
        
        print("self.mb_optim.param_groups[0][lr]", self.mb_optim.param_groups[0]["lr"])