import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math
from scipy.stats import entropy

from trainer import Trainer
from router import Router
from model_util import plot_attention_weights, window_and_pad
from data_loader import DataLoader
from block_trainer import BlockTrainer

action_sets = {}

class ClassifierTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.load_data()
        self._block_trainer = BlockTrainer(self.args, self._fb_model, self.fb_optim)
        self._action_smooth_loss = nn.KLDivLoss()
                
    def train(self):
        critic_losses = []
        model_bce_losses = []

        #self.train_data = self.train_data[:2048]
        #self.test_data = self.test_data[:512]
        
        for epoch in range(self.args.epochs):
            np.random.shuffle(self.train_data)
            for i in range(len(self.train_data)//self.args.batch_size):
                
                self._fb_count = torch.zeros(self._router.num_fbs + 1)
                #tasks = #np.vstack(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0])
                task_ids = torch.tensor(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0].astype(int)).cuda()
                # if self.args.constraints and int(task) not in self.tasks:
                #     continue
                #plt.imshow(self.train_data[i % len(self.train_data)][1][0].transpose(1,2,0))
                #plt.show()
                #self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                
                #print(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1].shape)
                imgs = np.vstack(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                imgs = torch.tensor(imgs).cuda()
                target = torch.tensor(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 2].astype(int), dtype=torch.int64).unsqueeze(1).cuda()

                router_results = self._router(imgs, task_ids)
                action_probs, actions, critic_outputs = router_results.action_probs, router_results.actions, router_results.critic_outputs


                # print("actions_mask", actions_mask)
                # print("actions_window", actions_window)
                # print("valid_fbs_ep_pad", valid_fbs_ep_pad)
                # Add reward bonus
                rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64).cuda()
                

                # Record the task architecture
                # if epoch > 50:
                # action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                # if action_task_pair not in action_sets:
                #     action_sets[action_task_pair] = 0
                # action_sets[action_task_pair] +=1
                
                # Train the output model
                _, output_losses = self._block_trainer.train(imgs, actions, target)
                
                # Reward based on model loss
                rewards[:, -1] += -output_losses
                
                if not self.args.no_reward_bonus:
                    self._fb_count += router_results.fb_count
                    
                    # 1. Compute average loss for every FB
                    # 2. Add pct. of average loss to reward
                    #rewards += output_losses / self._fb_count
                    output_losses_batch = torch.repeat_interleave(output_losses, (self.args.max_depth), axis=0)
                    bonus = self.layer_reward_bonus(actions, output_losses_batch)
                    # print("bonus", bonus)
                    # print("rewards", rewards)
                    # print("actions", actions)
                    # print("fb_count", self._fb_count)
                    rewards += bonus
                
                model_bce_losses.append(output_losses.tolist())
                
                returns, advs, tdlamret = self.compute_return(rewards, action_probs, critic_outputs)
                advs = torch.reshape(advs, (-1, ))
                tdlamret = torch.reshape(tdlamret, (-1, ))

                # Set up making each timestep an example
                actions_mask, actions_window, valid_fbs_ep_pad, actions_one_hot, action_probs_pad, valid_fbs_count, actions_batch = window_and_pad(
                    router_results.valid_fbs_ep,
                    actions,
                    router_results.action_indices,
                    action_probs,
                    self.args.max_depth,
                    router_results.max_fbs)
                
                imgs_batch_steps = torch.repeat_interleave(imgs, (self.args.max_depth), axis=0)
                task_ids_steps = torch.repeat_interleave(task_ids, (self.args.max_depth), axis=0)
                num_exs = self.args.max_depth * self.args.batch_size

                with torch.autograd.set_detect_anomaly(True):
                    total_critic_loss = 0
                    total_actor_loss = 0
                    for e_i in range(self.args.noptepochs):
                        batch_ids = torch.randperm(num_exs)
                        loss = torch.zeros(1).cuda()

                        total_smooth_loss = 0
                        
                        #advs = torch.clamp(advs, min=-3, max=3)
                        #critic_loss = self.critic_loss_fn(critic_outputs, tdlamret) * self.args.critic_lam
                        #print("CRITIC LOSS", critic_loss)
                        
                        # Fraction of probability allocated to layer over batch


                        # Fraction of tasks allocated to layer over batch
                        action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()

                        for m_i in range(self.args.nminibatches):
                            loss = torch.zeros(1).cuda()
                            start = m_i * (num_exs // self.args.nminibatches)
                            end = (m_i + 1) * (num_exs // self.args.nminibatches)
                            m_batch_idxs = batch_ids[start:end]
                            # Get prediction for current timestep
                            new_prob, val_pred = self._router.predict_step(
                                imgs_batch_steps[m_batch_idxs],
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
                                actions_one_hot[m_batch_idxs])

                            
                            total_critic_loss += self.args.critic_lam * critic_loss.detach().cpu().numpy()
                            total_actor_loss += actor_loss.detach().cpu().numpy()
                            loss += actor_loss + self.args.critic_lam * critic_loss
                            load_balance_loss =self.args.load_bal_lam * self.compute_load_balance_loss(valid_fbs_count[m_batch_idxs], valid_fbs_ep_pad[m_batch_idxs], new_prob, actions_batch[m_batch_idxs]) 
                            loss += load_balance_loss
                            if self.args.smooth_lam > 0.0:
                                smooth_loss = self.label_smooth_loss(router_results, new_prob, valid_fbs_count[m_batch_idxs], actions_one_hot[m_batch_idxs], m_batch_idxs)
                                loss += smooth_loss
                                total_smooth_loss += smooth_loss.detach().cpu().numpy()
                                
                            
                            

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self._router.parameters(), 2.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        # Compute loss
                        # for t in range(self.args.max_depth):
                            
                        #     #critic_loss = self.critic_loss_fn(critic_outputs[:, t], tdlamret[:, t].unsqueeze(-1)) * (self.args.max_depth) * self.args.critic_lam
                        #     #total_critic_loss += critic_loss.detach().cpu().numpy()
                            
                        #     #loss += critic_loss
                        #     #loss += -torch.log(action_probs[t][0][router_results.action_indices[t]]) * adv

                        #     # delta = advs[t] - self.adv_avg[t] 
                        #     # adv = (advs[t] - self.adv_avg[t]) / self.adv_std[t]**0.5
                        #     # adv = max(min(3.0, adv), -3.0)                        #         # Add label-smoothing loss
                        #         num_valid_fbs = len(router_results.valid_fbs_ep[b_i][t])
                        #         if num_valid_fbs > 1:
                        #             max_action_prob = (1-self.args.alpha) + self.args.alpha/num_valid_fbs
                        #             if action_prob > max_action_prob:
                        #                 smooth_actions = torch.zeros_like(action_probs[b_i][t])#torch.tensor(self.args.alpha).repeat_interleave(num_valid_fbs)
                        #                 # Add prob to all valid actions
                        #                 smooth_actions[:num_valid_fbs] = self.args.alpha/num_valid_fbs
                        #                 # Add extra prob to selected action
                        #                 smooth_actions[router_results.action_indices[b_i][t]] = max_action_prob
                        #                 smooth_loss = self.args.smooth_lam * self._action_smooth_loss(torch.log(action_probs[b_i][t] + 1e-7), smooth_actions)
                        #                 loss += smooth_loss
                        #                 total_smooth_loss += smooth_loss.detach().cpu().numpy()
                                
                            
                        #     # self.adv_std[t] = (1-self.args.adv_alpha) * (self.adv_std[t] + self.args.adv_alpha * delta**2) 
                        #     # self.adv_avg[t] = self.adv_avg[t] + self.args.adv_alpha * delta 
                            
                        #     for b_i in range(self.args.batch_size):
                        #         action_prob = action_probs[b_i][t][router_results.action_indices[b_i][t]]
                        #         if action_prob == 0:
                        #             print("action_probs", action_probs)
                        #             print("action_probs[b_i][t]", action_probs[b_i][t])
                        #             print(router_results.action_indices[b_i][t])
                        #             print(action_prob)
                        #             assert False
                                
                        #         # Add label-smoothing loss
                        #         num_valid_fbs = len(router_results.valid_fbs_ep[b_i][t])
                        #         if num_valid_fbs > 1:
                        #             max_action_prob = (1-self.args.alpha) + self.args.alpha/num_valid_fbs
                        #             if action_prob > max_action_prob:
                        #                 smooth_actions = torch.zeros_like(action_probs[b_i][t])#torch.tensor(self.args.alpha).repeat_interleave(num_valid_fbs)
                        #                 # Add prob to all valid actions
                        #                 smooth_actions[:num_valid_fbs] = self.args.alpha/num_valid_fbs
                        #                 # Add extra prob to selected action
                        #                 smooth_actions[router_results.action_indices[b_i][t]] = max_action_prob
                        #                 smooth_loss = self.args.smooth_lam * self._action_smooth_loss(torch.log(action_probs[b_i][t] + 1e-7), smooth_actions)
                        #                 loss += smooth_loss
                        #                 total_smooth_loss += smooth_loss.detach().cpu().numpy()
                                
                        #         # Get prediction for current timestep
                        #         new_prob, val_pred = self._router.predict_step(
                        #             imgs[b_i].unsqueeze(0),
                        #             task_ids[b_i].unsqueeze(0),
                        #             torch.tensor(router_results.valid_fbs_ep[b_i][t], dtype=torch.int64).cuda(),
                        #             torch.tensor(actions[b_i][:t], dtype=torch.int64).cuda())
                        #         actor_loss, critic_loss = self.ppo_loss(
                        #             advs[b_i][t], 
                        #             action_prob, 
                        #             new_prob[0][router_results.action_indices[b_i][t]], 
                        #             tdlamret[b_i][t].unsqueeze(-1).unsqueeze(-1), 
                        #             val_pred)
                        #         # print("action_prob", action_prob)
                        #         # print("new_prob[router_results.action_indices[b_i][t]]", new_prob[0][router_results.action_indices[b_i][t]]) 
                        #         # print("ACTOR LOSS, ", actor_loss)
                        #         # print("critic_loss", critic_loss)
                        #         actor_loss = actor_loss / self.args.batch_size
                        #         critic_loss = critic_loss / self.args.batch_size  
                        #         #actor_loss = -torch.log(action_prob) * advs[b_i][t] / self.args.batch_size 
                        #         loss += actor_loss + self.args.critic_lam * critic_loss 
                        #         total_actor_loss += actor_loss.detach().cpu().numpy()
                        #         total_critic_loss += critic_loss.detach().cpu().numpy()

                        #         action_count[actions[b_i][t]] += 1
                        #         for j, a_t in enumerate(router_results.valid_fbs_ep[b_i][t]):
                        #             action_prob_sum[a_t] += 1/(self.args.batch_size * self.args.max_depth) * action_probs[b_i][t][j]
                        #         #enp_loss = -self.args.entropy_lam * action_prob * torch.log(action_prob +1e-10) / self.args.batch_size
                        #         #total_enp_loss += enp_loss
                        #         #loss += enp_loss                       
                        #     # if not router_results.failed and action_probs[t][0][actions[t]] > 0.50 and action_probs[t][0][actions[t]] != 0:
                        #     #     loss += action_probs[t][0][actions[t]] * 10

                        #     # Add entropy regularization
                            


                        #loss += critic_loss
                        critic_losses.append(total_critic_loss)
                        #print("Entropy loss", -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10)))

                        # Update router model
                        # Load-balance loss


                if i % 2 == 0:
                    #print("Output BCE Losses", output_losses)
                    print("router_results.fb_count", router_results.fb_count)
                    print("\nEpoch:", epoch, "Iter:", i*self.args.batch_size, "\n")
                    # print("TOTAL ENP LOSS", total_enp_loss)
                    print("CRITIC LOSS", total_critic_loss)
                    print("TOTAL ACTOR LOSS", total_actor_loss)
                    print("load_balance_loss", load_balance_loss)
                    print("ACTION SMOOTHING LOSS", total_smooth_loss)
                    print("\ntask", task_ids[0], " target", target[0])
                    for j, action_prob in enumerate(action_probs[0]):
                        print(f"action_probs[{j}]", action_prob)
                    print("Critic Outputs", val_pred[:5].squeeze(-1).detach().cpu().numpy(), "VS", tdlamret[m_batch_idxs][:5].detach().cpu().numpy())
                    print("ADVS", advs[m_batch_idxs][:5].detach().cpu().numpy())
                    # print("Avg ADV", self.adv_avg)
                    # print("STD ADV", self.adv_std)
                    
                    # if epoch > 18 and i == 500:
                    #     state_labels = [["Task " + str(int(task)), "Encoded State"] + actions[:i] for i in range(len(actions))]
                    #     layer_ids = [list(set(range(action_probs[0].shape[1])).difference(set(actions[:i]))) for i in range(len(actions))]
                    #     plot_attention_weights(state_labels, layer_ids, srouter_results.att_weights)
                    print(actions[0], "\n")

            test_f1_score, test_acc_score, load_bal_loss, action_entropy, num_unused_fbs = self.test_models()
            self.test_results["test_f1_scores"].append(test_f1_score)
            self.test_results["test_acc_scores"].append(test_acc_score)
            self.test_results["load_bal_loss"].append(load_bal_loss)
            self.test_results["action_entropy"].append(action_entropy)
            self.test_results["num_unused_fbs"].append(num_unused_fbs)



            self.save()

        for pair in sorted(list(action_sets.items()), key=lambda a: a[1]):
            print(pair)
        


        # train_results = {
        #     "critic_losses" : critic_losses,
        #     "model_bce_losses" : model_bce_losses,
        #     "epochs" : self.args.epochs,
        #     "constraints" : self.args.constraints
        # }
            

        # fig, axes = plt.subplots(4)
        # axes[0].plot(critic_losses)
        # axes[0].set_title("Critic Loss")
        # axes[1].plot(model_bce_losses)
        # axes[1].set_title("BCE Loss")
        # axes[2].plot(test_f1_scores)
        # axes[2].set_title("Test F1 Score")

        # axes[3].plot(test_acc_scores)
        # axes[3].set_title("Test Accuracy")


        # plt.show()
        # self.save()
        
        
    # def compute_actor_bonus(self, new_prob):

    #     for i in range(batch_size):
    #         for j in range(depth):
    #             if actions[i][j] != PASS_FB



    def compute_load_balance_loss(self, valid_fbs_count, valid_fbs_ep_pad, new_prob, actions_batch):
        kl = torch.nn.KLDivLoss()
        action_prob_sum = torch.zeros(self._router.num_fbs + 1, 1).cuda()        
        batch_size = valid_fbs_ep_pad.shape[0]
        action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()
        #total_smooth_loss = torch.zeros(1).cuda()
        
        for b_i in range(batch_size):
            num_valid_fbs = valid_fbs_count[b_i].int()
            
            for j in range(num_valid_fbs):
                a_t = valid_fbs_ep_pad[b_i][j]
                action_prob_sum[a_t] += new_prob[b_i][j]
                action_count[a_t] += 1
        
        action_prob_sum = action_prob_sum / (action_count + 1)
        load_balance_loss = kl(torch.log(action_prob_sum[1:] + 1e-7), self._fb_model.target_prob)
        
        return load_balance_loss

    def load_data(self):
        self._data_loader = DataLoader(self.args)
        
        self.train_data, self.test_data = self._data_loader.load_mnist()

        # Load constraints if specified
        if self.args.load_constraints:
            with open(self.args.load_constraints) as f:
                constraints = json.load(f)
                self.args.constraints = {}
                for k,v in constraints.items():
                    self.args.constraints[int(k)] = v
                print(self.args.constraints)
                self.tasks = set(int(t) for t in self.args.constraints.keys())
        else:
            self.args.constraints = None


    def test_models(self):
        self._router.eval()
        self._fb_model.eval()
        with torch.no_grad():
            action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()
            action_prob_sum = torch.zeros(self._router.num_fbs + 1, 1).cuda()
                
            print("Testing models")
            #for k,v in self.args.constraints.items():
            y_trues = np.zeros(len(self.test_data))
            y_preds = np.zeros(len(self.test_data))
            np.random.shuffle(self.test_data)
            for i in range(math.ceil(len(self.test_data)/self.args.batch_size)): 
                task= torch.tensor(self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0].astype(int)).cuda()
                imgs = np.vstack(self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                imgs = torch.tensor(imgs).cuda()
                targets = self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 2]
                print("task[0]", task[0], "targets[0]", targets[0])
                y_trues[i*self.args.batch_size:(i+1)*self.args.batch_size] = targets
                router_results = self._router(imgs, task, argmax=True, should_drop_fbs=False)
                
                pred = self._fb_model(imgs, router_results.actions)

                if self.args.num_classes == 1:
                    pred = torch.sigmoid(pred)
                    class_preds = pred >= self.args.decision_threshold
                    class_preds = class_preds.squeeze(1).cpu().numpy().astype(int)
                else:
                    class_preds = torch.argmax(pred, dim=1)
                    class_preds = class_preds.cpu().numpy().astype(int)                
                y_preds[i*self.args.batch_size:(i+1)*self.args.batch_size] =class_preds 
                
                # Add to action count and prob
                print(router_results.action_probs[0])
                print(router_results.actions[0])

                for t in range(self.args.max_depth):
                    for b_i in range(task.shape[0]):
                        action_count[router_results.actions[b_i][t]] += 1
                        for j, a_t in enumerate(router_results.valid_fbs_ep[b_i][t]):
                            action_prob_sum[a_t] += 1/(self.args.batch_size * self.args.max_depth) * router_results.action_probs[b_i][t][j]
            
            if self.args.num_classes == 1:
                test_f1_score = f1_score(y_trues, y_preds)
                print("F1 score of testing dataset", test_f1_score)
                test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
                print("Accuracy of testing dataset", test_acc_score)
            else:
                test_f1_score = f1_score(y_trues, y_preds, average="macro")
                print("F1 score of testing dataset", test_f1_score)
                test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
                print("Accuracy of testing dataset", test_acc_score)
            load_bal_loss = 0#self.compute_load_balance_loss(action_prob_sum, action_count).cpu().tolist()[0]
            print("LOAD BALANCE LOSS:", load_bal_loss)
 
            action_prob = action_count / action_count.sum()
            action_entropy = float(entropy(action_prob.cpu())[0])
            num_unused_fbs = int(len(action_count) - torch.count_nonzero(action_count))
            print("NUM UNUSED FBS", num_unused_fbs)
            print("ACTION ENTROPY", action_entropy)
            print("ACTION COUNT: ", action_count)
            
            
            
        
        self._router.train()
        self._fb_model.train()
        return test_f1_score, test_acc_score, load_bal_loss, action_entropy, num_unused_fbs

    def ppo_loss(self, adv, old_prob, new_prob, val_true, val_pred, actions_one_hot):
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
        vf_loss = self.critic_loss_fn(val_pred.squeeze(-1), val_true)

        return  policy_loss, vf_loss

    def label_smooth_loss(self, router_results, new_prob, valid_fbs_count, actions_one_hot, m_batch_idxs):
        """"Add label-smoothing loss."""
        batch_size = new_prob.shape[0]
        total_smooth_loss = torch.zeros(1).cuda()
        action_probs = torch.sum(actions_one_hot * new_prob, axis=-1)
        
        b_i_old = m_batch_idxs // self.args.max_depth
        t_old = m_batch_idxs % self.args.max_depth
        for b_i in range(batch_size):
            num_valid_fbs = valid_fbs_count[b_i].int()
            if num_valid_fbs > 1:
                max_action_prob = (1-self.args.alpha) + self.args.alpha/num_valid_fbs
                if action_probs[b_i] > max_action_prob:
                    smooth_actions = torch.zeros_like(new_prob[b_i])
                    # Add prob to all valid actions
                    smooth_actions[:num_valid_fbs] = self.args.alpha/num_valid_fbs

                    
                    # Add extra prob to selected action
                    smooth_actions[router_results.action_indices[b_i_old[b_i]][t_old[b_i]]] = max_action_prob
                    
                    smooth_loss = self.args.smooth_lam * self._action_smooth_loss(torch.log(new_prob[b_i] + 1e-7), smooth_actions)
                    total_smooth_loss += smooth_loss

        return total_smooth_loss / batch_size
        