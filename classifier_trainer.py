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
from model_util import plot_attention_weights, window_and_pad, create_lookahead_mask
from data_loader import DataLoader
from block_trainer import BlockTrainer

action_sets = {}

class ClassifierTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.load_data()
        self._block_trainer = BlockTrainer(self.args, self._fb_model, self.fb_optim)

                
    def train(self):
        critic_losses = []
        model_bce_losses = []

        # np.random.shuffle(self.train_data)
        # np.random.shuffle(self.test_data)
        # self.train_data = self.train_data[:7000]
        # self.test_data = self.test_data[:1024]
        
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
                with torch.no_grad():
                    router_results = self._router(imgs, task_ids)
                action_probs, actions, critic_outputs = router_results.action_probs, router_results.actions, router_results.critic_outputs


                # Add reward bonus
                rewards = torch.zeros_like(actions[-1], dtype=torch.float64).cuda()

                # Train the output model
                _, output_losses = self._block_trainer.train(imgs, actions[-1], target)


                # Reward based on model loss
                rewards[:, -1] += -output_losses

                if not self.args.no_reward_bonus:
                    self._fb_count += router_results.fb_count
                    
                    bonus = self.layer_reward_bonus(actions[-1])
                    rewards += bonus
                
                model_bce_losses.append(output_losses.tolist())

                returns, advs, tdlamret = self.compute_return(rewards, critic_outputs)
                # advs = torch.reshape(advs, (-1, ))
                # tdlamret = torch.reshape(tdlamret, (-1, ))

                # Create masks for decoder
                # TODO: Update this to change shape and need to change predict_step critic reshaping as well
                fb_masks = create_lookahead_mask(self.args.max_depth).cuda()

                # Reshape so you can treat each timestep as an example
                action_masks = router_results.action_masks#.transpose(0,1)#router_results.action_masks.reshape(router_results.action_masks.shape[1],router_results.action_masks.shape[0], router_results.action_masks.shape[2])

                actions_final = actions[-1]
                
                actions_inp = torch.zeros(self.args.batch_size, self.args.max_depth, dtype=torch.int64).cuda()
                actions_inp[:, 1:] = actions[-1, :, :-1]
                

                
               
                critic_outputs = critic_outputs.reshape(-1,1)
                with torch.autograd.set_detect_anomaly(True):
                    total_critic_loss = 0
                    total_actor_loss = 0
                    for e_i in range(self.args.noptepochs):
                        batch_ids = torch.arange(self.args.batch_size)
                        loss = torch.zeros(1).cuda()

                        total_smooth_loss = 0

                        # Fraction of tasks allocated to layer over batch
                        action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()

                        for m_i in range(self.args.nminibatches):
                            loss = torch.zeros(1).cuda()
                            start = m_i * (self.args.batch_size // self.args.nminibatches)
                            end = (m_i + 1) * (self.args.batch_size // self.args.nminibatches)
                            m_batch_idxs = batch_ids[start:end]

                            # Get prediction for current timestep
                            new_prob, val_pred = self._router.predict_step(
                                imgs[m_batch_idxs],
                                task_ids[m_batch_idxs],
                                actions_inp[m_batch_idxs],
                                fb_masks,
                                action_masks[m_batch_idxs])


                            actor_loss, critic_loss = self.ppo_loss(
                                advs[m_batch_idxs].view(-1), 
                                action_probs[m_batch_idxs].reshape(-1,action_probs.shape[2]), 
                                new_prob, 
                                tdlamret[m_batch_idxs].view(-1), 
                                val_pred.view(-1),
                                actions_final[m_batch_idxs].flatten().unsqueeze(1))
                            
                            # # TODO: DELETE THIS
                            # import sys
                            # sys.exit()

                            total_critic_loss += self.args.critic_lam * critic_loss.detach().cpu().numpy()
                            total_actor_loss += actor_loss.detach().cpu().numpy()
                            loss += actor_loss + self.args.critic_lam * critic_loss
                            load_balance_loss = 0
                            # if self.args.load_bal_lam > 0:
                            #     load_balance_loss =self.args.load_bal_lam * self.compute_load_balance_loss(valid_fbs_count[m_batch_idxs], valid_fbs_ep_pad[m_batch_idxs], new_prob, actions_batch[m_batch_idxs]) 
                            #     loss += load_balance_loss
                            # else:
                            #     load_balance_loss = 0
                            # if self.args.smooth_lam > 0.0:
                            #     smooth_loss = self.label_smooth_loss(router_results, new_prob, valid_fbs_count[m_batch_idxs], actions_one_hot[m_batch_idxs], m_batch_idxs)
                            #     loss += smooth_loss
                            #     total_smooth_loss += smooth_loss.detach().cpu().numpy()                            

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self._router.parameters(), self.args.max_grad_norm)
                      
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        critic_losses.append(total_critic_loss)
                        
                        #print("Entropy loss", -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10)))

                        # Update router model
                        # Load-balance loss


                if i % 16 == 0:
                    #print("Output BCE Losses", output_losses)
                    print("router_results.fb_count", router_results.fb_count)
                    print("\nEpoch:", epoch, "Iter:", i*self.args.batch_size, "\n")
                    # print("TOTAL ENP LOSS", total_enp_loss)
                    print("CRITIC LOSS", total_critic_loss)
                    print("TOTAL ACTOR LOSS", total_actor_loss)
                    # print("load_balance_loss", load_balance_loss)
                    # print("ACTION SMOOTHING LOSS", total_smooth_loss)
                    print("\ntask", task_ids[0], " target", target[0])
                    # for j, action_prob in enumerate(action_probs[0]):
                    #     print(f"action_probs[{j}]", action_prob)
                    print("Critic Outputs", val_pred[:5].squeeze(-1).detach().cpu().numpy(), "VS", tdlamret[m_batch_idxs][:5].detach().cpu().numpy())
                    print("ADVS", advs[m_batch_idxs][:5].detach().cpu().numpy())
                    # print("Avg ADV", self.adv_avg)
                    # print("STD ADV", self.adv_std)
                    
                    # if epoch > 18 and i == 500:
                    #     state_labels = [["Task " + str(int(task)), "Encoded State"] + actions[:i] for i in range(len(actions))]
                    #     layer_ids = [list(set(range(action_probs[0].shape[1])).difference(set(actions[:i]))) for i in range(len(actions))]
                    #     plot_attention_weights(state_labels, layer_ids, srouter_results.att_weights)
                    print(router_results.actions[-1, -1], "\n")
                    print("router_results.action_probs[0, 0]", router_results.action_probs[0, 0], task_ids[0])
                    print("router_results.action_probs[0, 1]", router_results.action_probs[0, 1], task_ids[1])


            test_f1_score, test_acc_score, load_bal_loss, action_entropy, num_unused_fbs = self.test_models()
            self.test_results["test_f1_scores"].append(test_f1_score)
            self.test_results["test_acc_scores"].append(test_acc_score)
            self.test_results["load_bal_loss"].append(load_bal_loss)
            self.test_results["action_entropy"].append(action_entropy)
            self.test_results["num_unused_fbs"].append(num_unused_fbs)



            self.save()

            # Decrease learning rate
            self._lr_scheduler.step()
            self._fb_lr_scheduler.step()
            for param_group in self.fb_optim.param_groups:
                print(param_group["lr"])
            #print("self.optimizer.lr", self._optimizer.lr)
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
                # print("task[0]", task[0], "targets[0]", targets[0])
                y_trues[i*self.args.batch_size:(i+1)*self.args.batch_size] = targets
                router_results = self._router(imgs, task, argmax=True, should_drop_fbs=False)
                
                pred = self._fb_model(imgs, router_results.actions[-1])

                if self.args.num_classes == 1:
                    pred = torch.sigmoid(pred)
                    class_preds = pred >= self.args.decision_threshold
                    class_preds = class_preds.squeeze(1).cpu().numpy().astype(int)
                else:
                    class_preds = torch.argmax(pred, dim=1)
                    class_preds = class_preds.cpu().numpy().astype(int)                
                y_preds[i*self.args.batch_size:(i+1)*self.args.batch_size] =class_preds 
                
                # Add to action count and prob
                for t in range(self.args.max_depth):
                    for b_i in range(task.shape[0]):
                        # print("router_results.actions[-1, b_i, t]", router_results.actions[-1, b_i, t])
                        action_count[router_results.actions[-1, b_i, t]] += 1
                        # for j, a_t in enumerate(router_results.valid_fbs_ep[b_i][t]):
                        #     action_prob_sum[a_t] += 1/(self.args.batch_size * self.args.max_depth) * router_results.action_probs[b_i][t][j]
            
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
            print("router_results.fb_count", router_results.fb_count)
            print("router_results.actions[-1, :5]", router_results.actions[-1, :5], "\n")
            print("router_results.action_probs[0, :5]", router_results.action_probs[0, :5])
            print("router_results.action_probs[0, :5]", router_results.action_probs[0, :5])
            
        
        self._router.train()
        self._fb_model.train()
        return test_f1_score, test_acc_score, load_bal_loss, action_entropy, num_unused_fbs

    # def ppo_loss(self, adv, old_prob, new_prob, val_true, val_pred, actions_one_hot):
    #     """PPO discreate loss function.
    #     Args:
    #         advantage: advantage from the episode.
    #         old_pred: old softmax predictions.
    #         y_true: one-hot encoded actions taken in the episode.
    #         y_pred: new softmax predictions.
    #         val_true: true values (i.e., returns) from the episode.
    #         val_pred: critic value predictions
        
    #     Returns:
    #         A tuple of PPO loss and value MSE loss 
    #     """
    #     new_prob = torch.log(torch.sum(actions_one_hot * new_prob, axis=-1))
    #     old_prob = torch.log(torch.sum(actions_one_hot * old_prob, axis=-1))

    #     # Compute ratio (NOTE: this is same as ratio = new_prob/old_prob) 
    #     ratio = torch.exp(new_prob - old_prob.detach())

    #     surrogate_1 = ratio * adv

    #     # Clipped policy gradient, ratio clipped between [0.8, 1.2]
    #     surrogate_2 = torch.clip(ratio, 1.0 - self.args.eps_clip, 1.0 + self.args.eps_clip) * adv

    #     # Entropy regularization
    #     #entropy_pen = -ENTROPY_COEF * tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-10), axis=1)
    #     entropy_pen = 0
    #     # PPO loss, averaged over several batches
    #     policy_loss = -torch.mean(torch.min(surrogate_1, surrogate_2))

    #     # Value loss (MSE)
    #     vf_loss = self.critic_loss_fn(val_pred.squeeze(-1), val_true)

    #     return  policy_loss, vf_loss


        
