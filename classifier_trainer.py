import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import math

from trainer import Trainer
from router import Router
from model_util import plot_attention_weights
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
        test_f1_scores = []
        test_acc_scores = []

        # Fraction of probability allocated to layer over batch
        action_prob_sum = torch.zeros(self._router.num_fbs + 1, 1).cuda()

        # Fraction of tasks allocated to layer over batch
        action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()
        
        #print("len(self.train_data)", len(self.train_data))
        #self.train_data = self.train_data[:128]
        # self.test_data = self.test_data[:256]

        for epoch in range(self.args.epochs):
            np.random.shuffle(self.train_data)
            for i in range(len(self.train_data)//self.args.batch_size):
                if i % 32 == 0:
                    print("\nEpoch:", epoch, "Iter:", i*self.args.batch_size, "\n")

                #tasks = #np.vstack(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0])
                task = torch.tensor(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0].astype(int)).cuda()
                # if self.args.constraints and int(task) not in self.tasks:
                #     continue
                #plt.imshow(self.train_data[i % len(self.train_data)][1][0].transpose(1,2,0))
                #plt.show()
                #self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                
                #print(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1].shape)
                imgs = np.vstack(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                img = torch.tensor(imgs).cuda()
                target = torch.tensor(self.train_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 2].astype(int), dtype=torch.float32).unsqueeze(1).cuda()

                router_results = self._router(img, task)
                action_probs, actions, critic_outputs = router_results.action_probs, router_results.actions, router_results.critic_outputs

                # Add reward bonus
                rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64).cuda()
                
                if not self.args.no_reward_bonus:
                    rewards += self.layer_reward_bonus(actions)
                # Record the task architecture
                # if epoch > 50:
                # action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                # if action_task_pair not in action_sets:
                #     action_sets[action_task_pair] = 0
                # action_sets[action_task_pair] +=1
                
                # Train the output model
                _, output_losses = self._block_trainer.train(img, actions, target)
                
                
                # Reward based on model loss
                rewards[:, -1] += -output_losses

                model_bce_losses.append(output_losses.tolist())

                
                returns, advs, tdlamret = self.compute_return(rewards, action_probs, critic_outputs)
                #returns[:len(added_layers)] += self.layer_reward_bonus(added_layers) 
                loss = torch.zeros(1).cuda()
                total_critic_loss = 0
                total_actor_loss = 0
                advs = (advs - advs.mean()) / advs.std()
                #advs = torch.clamp(advs, min=-3, max=3)
                #critic_loss = self.critic_loss_fn(critic_outputs, tdlamret) * self.args.critic_lam
                #print("CRITIC LOSS", critic_loss)
                
                
                # Compute loss
                for t in range(len(action_probs[0])):
                    #critic_loss =  * (1/len(critic_outputs)) * self.args.critic_lam 
                    critic_loss = self.critic_loss_fn(critic_outputs[:, t], tdlamret[:, t].unsqueeze(-1)) * (self.args.max_depth) * self.args.critic_lam
                    total_critic_loss += critic_loss.detach().cpu().numpy()
                    loss += critic_loss
                    #loss += -torch.log(action_probs[t][0][router_results.action_indices[t]]) * adv

                    # delta = advs[t] - self.adv_avg[t] 
                    # adv = (advs[t] - self.adv_avg[t]) / self.adv_std[t]**0.5
                    # adv = max(min(3.0, adv), -3.0)
                    
                    # self.adv_std[t] = (1-self.args.adv_alpha) * (self.adv_std[t] + self.args.adv_alpha * delta**2) 
                    # self.adv_avg[t] = self.adv_avg[t] + self.args.adv_alpha * delta 
                    
                    # if router_results.failed and t + 1 == len(action_probs) and adv >= 0:
                    #     adv = -3
                        # returns[t] = -3
                    
                    for b_i in range(self.args.batch_size):
                        action_prob = action_probs[b_i][t][router_results.action_indices[b_i][t]]
                        if action_prob == 0:
                            print("action_probs", action_probs)
                            print("action_probs[b_i][t]", action_probs[b_i][t])
                            print(router_results.action_indices[b_i][t])
                            print(action_prob)
                            assert False
                        actor_loss = -torch.log(action_prob) * advs[b_i][t] / self.args.batch_size 
                        loss += actor_loss
                        total_actor_loss += actor_loss.detach().cpu().numpy()
                        action_count[actions[b_i][t]] += 1/(self.args.batch_size * self.args.max_depth)
                        
                        for i, a_t in enumerate(router_results.valid_fbs_ep[b_i][t]):
                            action_prob_sum[a_t] += 1/(self.args.batch_size * self.args.max_depth) * action_probs[b_i][t][i] 
                    # if not router_results.failed and action_probs[t][0][actions[t]] > 0.50 and action_probs[t][0][actions[t]] != 0:
                    #     loss += action_probs[t][0][actions[t]] * 10

                    # Add entropy regularization
                    #loss += -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10))


                #loss += critic_loss
                critic_losses.append(total_critic_loss)
                #print("Entropy loss", -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10)))

                # Update router model
                # Load-balance loss
                load_balance_loss = torch.zeros(1).cuda()
                for j in range(len(action_prob_sum)):
                    load_balance_loss +=  action_prob_sum[j] * action_count[j]

                load_balance_loss = self.args.load_bal_lam * (self._router.num_fbs * load_balance_loss - 1/self._router.num_fbs)
                loss += load_balance_loss
                loss.backward()

                # print("\nself._router.fb_model._fbs[0].layer.weight.grad.max()", self._router.fb_model._fbs[0].layer.weight.grad.max())
                # print("self._router.fb_model._fbs[0].layer.weight.grad.min()", self._router.fb_model._fbs[0].layer.weight.grad.min())
                # print("self._router.critic_out.weight.grad.max()", self._router.critic_out.weight.grad.max())
                # print("self._router.critic_out.weight.grad.min()", self._router.critic_out.weight.grad.min())
                # print("self._router.task_embs.weight.grad.min()", self._router.task_embs.weight.grad.min())
                # print("self._router.task_embs.weight.grad.max()", self._router.task_embs.weight.grad.max())
                # print("self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.max()", self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.max())
                # print("self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.min()", self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.min())
                # print("self._router.trans_router.decoder._layers[0].pw_ffn.fc1.weight.grad.max()", self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.max())
                # print("self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.min()", self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.min())

                torch.nn.utils.clip_grad_norm_(self._router.parameters(), 2.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Fraction of probability allocated to layer over batch
                action_prob_sum = torch.zeros(self._router.num_fbs + 1, 1).cuda()

                # Fraction of tasks allocated to layer over batch
                action_count = torch.zeros(self._router.num_fbs + 1, 1).cuda()

                if i % 8 == 0:
                    print("CRITIC LOSS", total_critic_loss)
                    print("TOTAL ACTOR LOSS", total_actor_loss)
                    print("load_balance_loss", load_balance_loss)
                    print("\ntask", task[0], " target", target[0])
                    for i, action_prob in enumerate(action_probs[0]):
                        print(f"action_probs[{i}]", action_prob)
                    print("Critic Outputs", critic_outputs[0].squeeze(-1).detach().cpu().numpy(), "VS", tdlamret.detach().cpu().numpy()[0], advs.detach().cpu().numpy()[0])
                    # print("Avg ADV", self.adv_avg)
                    # print("STD ADV", self.adv_std)
                    
                    # if epoch > 18 and i == 500:
                    #     state_labels = [["Task " + str(int(task)), "Encoded State"] + actions[:i] for i in range(len(actions))]
                    #     layer_ids = [list(set(range(action_probs[0].shape[1])).difference(set(actions[:i]))) for i in range(len(actions))]
                    #     plot_attention_weights(state_labels, layer_ids, srouter_results.att_weights)
                    print(actions[0], "\n")

            self.save()
            test_f1_score, test_acc_score = self.test_models()
            test_f1_scores.append(test_f1_score)
            test_acc_scores.append(test_acc_score)


        # test_f1_score, test_acc_score = self.test_models()
        # test_f1_scores.append(test_f1_score)
        # test_acc_scores.append(test_acc_score)

        for pair in sorted(list(action_sets.items()), key=lambda a: a[1]):
            print(pair)
        
        train_results = {
            "critic_losses" : critic_losses,
            "model_bce_losses" : model_bce_losses,
            "test_f1_scores" :test_f1_scores,
            "test_acc_scores" : test_acc_scores,
            "epochs" : self.args.epochs,
            "constraints" : self.args.constraints
        }
            

        fig, axes = plt.subplots(4)
        axes[0].plot(critic_losses)
        axes[0].set_title("Critic Loss")
        axes[1].plot(model_bce_losses)
        axes[1].set_title("BCE Loss")
        axes[2].plot(test_f1_scores)
        axes[2].set_title("Test F1 Score")

        axes[3].plot(test_acc_scores)
        axes[3].set_title("Test Accuracy")


        plt.show()
        self.save()
        self.save_results(train_results)
        

    def load_balance_loss(self):
        # Max over actions
        #(torch.tensor(action_probs))
        # Keep count of actions per layer
        # When Batch is done:
        #   Compute action distribution for each layer
        #   PROBLEM: the distribution will not reflect what layers are actually possiblly
        pass

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
        with torch.no_grad():
            print("Testing models")
            #for k,v in self.args.constraints.items():
            y_trues = np.zeros(len(self.test_data))
            y_preds = np.zeros(len(self.test_data))
            np.random.shuffle(self.test_data)
            for i in range(math.ceil(len(self.test_data)/self.args.batch_size)): 
                task = torch.tensor(self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 0].astype(int)).cuda()
                imgs = np.vstack(self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 1])
                imgs = torch.tensor(imgs).cuda()
                targets = self.test_data[i*self.args.batch_size:(i+1)*self.args.batch_size, 2]
                y_trues[i*self.args.batch_size:(i+1)*self.args.batch_size] = targets
                #failed, model = self._router.build_fixed(v)
                router_results = self._router(imgs, task, argmax=True, should_drop_fbs=False)

                # print("TASK:", task[0], "TARGETS", targets[0])
                # print(router_results.actions[0])
                # print(router_results.action_probs[0])
                # for a_t in router_results.actions[0]:
                #     print(a_t)
                #     print(self._fb_model.get_fb(a_t))

                # plt.imshow(imgs[0].cpu().numpy().transpose(1,2,0) * 0.3081 - 0.1307, cmap="gray")
                # plt.show()

                pred = torch.sigmoid(self._fb_model(imgs, router_results.actions))
                class_preds = pred >= self.args.decision_threshold
                y_preds[i*self.args.batch_size:(i+1)*self.args.batch_size] = class_preds.squeeze(1).cpu().numpy().astype(int)
                            
            test_f1_score = f1_score(y_trues, y_preds)
            print("F1 score of testing dataset", test_f1_score)
            test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
            print("Accuracy of testing dataset", test_acc_score)

        return test_f1_score, test_acc_score
