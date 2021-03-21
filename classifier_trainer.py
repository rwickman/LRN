import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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
        self._block_trainer = BlockTrainer(self.args)
                
    def train(self):
        total_loss = None
        critic_losses = []
        model_bce_losses = []
        test_f1_scores = []
        test_acc_scores = []

        # Fraction of probability allocated to layer over batch
        action_prob_sum = torch.zeros(self._router.num_layers + 1, 1).cuda()

        # Fraction of tasks allocated to layer over batch
        action_count = torch.zeros(self._router.num_layers + 1, 1).cuda()
        
        #print("len(self.train_data)", len(self.train_data))
        # self.train_data = self.train_data[:1024]
        # self.test_data = self.test_data[:512]

        for epoch in range(self.args.epochs):
            np.random.shuffle(self.train_data)
            for i, ex in enumerate(self.train_data):
                if i % 200 == 0:
                    print("\nEpoch:", epoch, "Iter:", i, "\n")
                #print(examples[0][i%self.args.batch_size:i%self.args.batch_size+1].shape)
                task = torch.tensor([ex[0]]).cuda()
                
                # if self.args.constraints and int(task) not in self.tasks:
                #     continue
                #plt.imshow(self.train_data[i % len(self.train_data)][1][0].transpose(1,2,0))
                #plt.show()
                img = torch.tensor(ex[1]).cuda()
                target = torch.tensor(ex[2], dtype=torch.float32).unsqueeze(0).cuda()

                router_result = self._router(img, task)
                action_probs, actions, model, critic_outputs = router_result.action_probs, router_result.actions, router_result.model, router_result.critic_outputs

                # Add reward bonus
                rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64).cuda()
                
                # TODO: FIX THIS
                if not self.args.no_reward_bonus:
                    bonus_mask = torch.tensor(actions) != 0
                    bonus_mask[-1] = not router_result.failed
                    rewards[bonus_mask] += self.layer_reward_bonus(actions, bonus_mask)

                # Record the task architecture
                # if epoch > 50:
                action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                if action_task_pair not in action_sets:
                    action_sets[action_task_pair] = 0
                action_sets[action_task_pair] +=1
                
                # Train the output model
                _, output_loss = self._block_trainer.train(model, img, target  )
                
                
                # Reward based on model loss
                rewards[-1] += -output_loss
                
                # Reward based on decision only
                # class_pred = int(torch.sigmoid(pred[0]) >= self.args.decision_threshold) 
                # print(class_pred == target.cpu().numpy()[0], class_pred, target.cpu().numpy()[0])
                # print(class_pred == target)
                # if class_pred == target.cpu().numpy()[0]:
                    # 
                #     rewards[-1] += 1
                # else:
                #     rewards[-1] += -1

                model_bce_losses.append(float(output_loss))

                
                returns, advs, tdlamret = self.compute_return(rewards, action_probs, critic_outputs)
                #returns[:len(added_layers)] += self.layer_reward_bonus(added_layers) 
                loss = torch.zeros(1).cuda()
                advantage = torch.zeros_like(returns)
                total_critic_loss = 0

                # Compute loss
                for t in range(len(action_probs)):
                    critic_loss = self.critic_loss_fn(critic_outputs[t].squeeze(0), tdlamret[t]) * (1/len(critic_outputs)) * self.args.critic_lam 

                    loss += critic_loss
                    total_critic_loss += critic_loss.detach().cpu().numpy()
                    
                    #adv = max(min(3.0, advs[t]), -3.0)
                    #adv = max(min(3.0, advs[t]), -3.0)
                    #loss += -torch.log(action_probs[t][0][router_result.action_indices[t]]) * adv

                    delta = advs[t] - self.adv_avg[t] 
                    #adv = (delta) / self.adv_std[t]**0.5
                    adv = (advs[t] - self.adv_avg[t]) / self.adv_std[t]**0.5
                    adv = max(min(3.0, adv), -3.0)
                    
                    self.adv_std[t] = (1-self.args.adv_alpha) * (self.adv_std[t] + self.args.adv_alpha * delta**2) 
                    self.adv_avg[t] = self.adv_avg[t] + self.args.adv_alpha * delta 
                    
                    # if router_result.failed and t + 1 == len(action_probs) and adv >= 0:
                    #     adv = -3
                        # returns[t] = -3
                    loss += -torch.log(action_probs[t][0][router_result.action_indices[t]]) * adv
                    advantage[t] = adv

                    # if not router_result.failed and action_probs[t][0][actions[t]] > 0.50 and action_probs[t][0][actions[t]] != 0:
                    #     loss += action_probs[t][0][actions[t]] * 10

                    # Add entropy regularization
                    loss += -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10))

                    # Add load-balancing loss
                    action_prob_sum[actions[t]] += 1/(self.args.batch_size * self.args.max_depth) * action_probs[t][0][router_result.action_indices[t]].clone()
                    action_count[actions[t]] += 1/(self.args.batch_size * self.args.max_depth) 
                    
                    #print("ACTOR LOSS", -torch.log(action_probs[t][0][router_result.action_indices[t]]) * adv)
                    #load_balance_loss = 1000 * self._router.num_layers * (1/(self.args.batch_size * self.args.max_depth)**2) * action_probs[t][0][router_result.action_indices[t]]
                    #print("LOAD BALANCE LOSS", load_balance_loss)
                    #loss += load_balance_loss

                
                critic_losses.append(total_critic_loss)
                #print("Entropy loss", -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10)))

                # Update router model
                loss = loss / self.args.batch_size
                loss.backward(retain_graph=True)
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
                
                # print("loss", loss)
                # print("total_loss", total_loss)
                
                if (i + 1) % self.args.batch_size == 0:

                    # Load-balance loss
                    load_balance_loss = torch.zeros(1).cuda()
                    for j in range(len(action_prob_sum)):
                         load_balance_loss += self.args.load_bal_lam * self._router.num_layers * action_prob_sum[j] * action_count[j]
                    load_balance_loss.backward()
                    print("load_balance_loss", load_balance_loss)
                    #print(action_prob_sum)
                    total_loss += load_balance_loss
                    print("total_loss", total_loss)
                    
                    #print(self._router.state_encoder.fc1.weight.grad)
                    #print("self._router.state_encoder.fc1.weight.grad.max()", self._router.state_encoder.fc1.weight.grad.max())
                    # print("self._router.critic_out.weight.grad.max()", self._router.critic_out.weight.grad.max())
                    # print("self._router.task_embs.weight.grad.max()", self._router.task_embs.weight.grad.max())
                    #print("self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.max()", self._router.trans_router.encoder.enc_layers[0].pw_ffn.fc1.weight.grad.max())
                    
                    #print("self._router.att.state_W.weight.grad.max()", self._router.att.state_W.weight.grad.max())
                    #print("self._router.att.state_W.weight.grad.max()", self._router.att.W_v.weight.grad.max())
                    #print("self._router.task_embs.weight.grad", self._router.task_embs.weight.grad)
                    # print("self._router.task_embs.weight.grad.norm()", self._router.task_embs.weight.grad.norm())
                    # print("self._router.task_embs.weight.grad.max()", self._router.task_embs.weight.grad.max())

                    torch.nn.utils.clip_grad_norm_(self._router.parameters(), 2.0)
                    #print("self._router.critic_out.weight.grad", self._router.critic_out.weight.grad)
                    #print("self._router.state_encoder.fc1.weight.grad", self._router.state_encoder.fc1.weight.grad)
                    #print("self._router.state_encoder.fc1.weight.grad", self._router.state_encoder.fc1.weight.grad.max())
                    self.optimizer.step()
                    total_loss = None
                    self.optimizer.zero_grad()
                    # Fraction of probability allocated to layer over batch
                    action_prob_sum = torch.zeros(self._router.num_layers + 1, 1).cuda()

                    # Fraction of tasks allocated to layer over batch
                    action_count = torch.zeros(self._router.num_layers + 1, 1).cuda()

                if i % 256 == 0:
                    print("\ntask", task, " target", target)
                    for i, action_prob in enumerate(action_probs):
                        print(f"action_probs[{i}]", action_prob)
                    print("Critic Outputs", torch.tensor(critic_outputs).cpu().numpy(), "VS", tdlamret.cpu().numpy(), advantage.detach().cpu().numpy())
                    print("Avg ADV", self.adv_avg)
                    print("STD ADV", self.adv_std)
                    
                    # if epoch > 18 and i == 500:
                    #     state_labels = [["Task " + str(int(task)), "Encoded State"] + actions[:i] for i in range(len(actions))]
                    #     layer_ids = [list(set(range(action_probs[0].shape[1])).difference(set(actions[:i]))) for i in range(len(actions))]
                    #     plot_attention_weights(state_labels, layer_ids, router_result.att_weights)

                    print(router_result.failed, model, actions, "\n")

            test_f1_score, test_acc_score = self.test_models()
            test_f1_scores.append(test_f1_score)
            test_acc_scores.append(test_acc_score)

            self.save()
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
        print("Testing models")
        # build model based on constraints only without router
        models = {}
        #for k,v in self.args.constraints.items():
        y_trues = np.zeros(len(self.test_data))
        y_preds = np.zeros(len(self.test_data))
        for i, test_ex in enumerate(self.test_data): 
            task = torch.tensor([test_ex[0]]).cuda()
            img = torch.tensor(test_ex[1]).cuda()
            y_trues[i] = test_ex[2]
            #failed, model = self._router.build_fixed(v)
            router_result = self._router(img, task, argmax=True, should_drop_layers=False)
            # print(router_result.action_probs)
            # print("TASK:", task)
            # print(router_result.model)
            # print(router_result.actions)
            # print(test_ex[1][0].max())
            # print(test_ex[1][0].min())

            # plt.imshow(test_ex[1][0].transpose(1,2,0) * 0.3081 - 0.1307, cmap="gray")
            # plt.show()
            
            pred = torch.sigmoid(router_result.model(img))
            y_preds[i] = int(pred >= self.args.decision_threshold)
                        
            # # Verify that the given constraints are valid
            # assert not failed

            # models[k] = model
        
        # Calculate the accuracy

        # for i, ex in enumerate(self.test_data):
        #     task = ex[0]
        #     img = torch.tensor(ex[1]).cuda()
            
        #     # Get predicted binary classification
        #     pred_model = models[task](img)
        #     pred = torch.sigmoid(models[task](img))
        #     y_preds[i] = int(pred >= self.args.decision_threshold)
        #     y_trues[i]  = ex[2]

        test_f1_score = f1_score(y_trues, y_preds)
        print("F1 score of testing dataset", test_f1_score)
        test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
        print("Accuracy of testing dataset", test_acc_score)

        return test_f1_score, test_acc_score