import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from router import Router
from sklearn.metrics import f1_score
from model_util import plot_attention_weights


bce_loss_fn = nn.BCEWithLogitsLoss()

action_sets = {}

class Trainer:
    def __init__(self, args):
        self.args = args
        self._router = Router(args, 10).cuda()
        
        self.load_data()
        
        #print(self._router.parameters())
        self.optimizer = optim.RMSprop(self._router.parameters(), lr=self.args.lr)#, betas=(0.5, 0.59))
        self.adv_avg = torch.zeros(self.args.max_depth).cuda()
        self.adv_std = torch.ones(self.args.max_depth).cuda()
        self.critic_loss_fn = nn.SmoothL1Loss()

        self._layer_count = {}
        if self.args.load:
            self.load()
        
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
        
        self.train_data = self.train_data
        print("len(self.train_data)", len(self.train_data))
        self.train_data = self.train_data[:2048]
        action_probs_batch = []
        action_indices_batch = [] 
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
                action_probs_batch.append(action_probs)
                action_indices_batch.append(router_result.action_indices)

                # Add reward bonus
                rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64).cuda()
                if not self.args.no_reward_bonus and len(actions) > 0:
                    bonus_mask = torch.tensor(actions) != 0
                    bonus_mask[-1] = not router_result.failed
                    rewards[bonus_mask] += self.layer_reward_bonus(actions, bonus_mask)

                if router_result.failed:
                    print(model)
                    assert not router_result.failed
                # Run through output network
                # Record the task architecture
                # if epoch > 50:
                action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                if action_task_pair not in action_sets:
                    action_sets[action_task_pair] = 0
                action_sets[action_task_pair] +=1
        
                model_optim = optim.SGD(model.parameters(), lr=1e-3)
                pred = model(img)

                loss = bce_loss_fn(pred[0], target)
                model_bce_losses.append(loss.detach().cpu().numpy())
                #print("MODEL LOSS: ", loss, "pred", torch.sigmoid(pred)[0].detach().cpu().numpy()[0], "vs", target.cpu().numpy()[0])

                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                rewards[-1] += -loss.detach().cpu().numpy()
                # print("rewards", rewards)

                
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
                    #loss += -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10))

                    # Add load-balancing loss
                    action_prob_sum[actions[t]] += 1/(self.args.batch_size * self.args.max_depth) * action_probs[t][0][router_result.action_indices[t]]
                    action_count[actions[t]] += 1/(self.args.batch_size * self.args.max_depth) 
                    
                    #print("ACTOR LOSS", -torch.log(action_probs[t][0][router_result.action_indices[t]]) * adv)
                    #load_balance_loss = 1000 * self._router.num_layers * (1/(self.args.batch_size * self.args.max_depth)**2) * action_probs[t][0][router_result.action_indices[t]]
                    #print("LOAD BALANCE LOSS", load_balance_loss)
                    #loss += load_balance_loss

                
                critic_losses.append(total_critic_loss)
                #print("Entropy loss", -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10)))

                # Update router model
                loss.backward()
                if total_loss is None:
                    total_loss = loss / self.args.batch_size
                else:
                    total_loss += loss / self.args.batch_size
                
                # print("loss", loss)
                # print("total_loss", total_loss)
                
                if (i + 1) % self.args.batch_size == 0:

                    # Load-balance loss
                    load_balance_loss = torch.zeros(1).cuda()
                    for j in range(len(action_prob_sum)):
                         load_balance_loss += self.args.load_bal_lam * self._router.num_layers * action_prob_sum[j] * action_count[j]
                    print("load_balance_loss", load_balance_loss)
                    print("total_loss", total_loss)
                    total_loss += load_balance_loss
                    # for i in range(len(action_probs_batch)):
                    #     for t in range(len(action_probs)):
                    #         total_loss += action_probs_batch[i][t][0][action_indices_batch[i][t]] / self.args.batch_size 
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

                if i % 100 == 0:
                    print("\ntask", task, " target", target)
                    print("action_probs[0]", action_probs[0])
                    if len(action_probs) > 1:
                        print("action_probs[1]", action_probs[1])
                    if len(action_probs) > 2:
                        print("action_probs[2]", action_probs[2])
                    print("Critic Outputs", torch.tensor(critic_outputs).cpu().numpy(), "VS", tdlamret.cpu().numpy(), advantage.detach().cpu().numpy())
                    print("Avg ADV", self.adv_avg)
                    print("STD ADV", self.adv_std)
                    
                    # if epoch > 18 and i == 500:
                    #     state_labels = [["Task " + str(int(task)), "Encoded State"] + actions[:i] for i in range(len(actions))]
                    #     layer_ids = [list(set(range(action_probs[0].shape[1])).difference(set(actions[:i]))) for i in range(len(actions))]
                    #     plot_attention_weights(state_labels, layer_ids, router_result.att_weights)

                    print(router_result.failed, model, actions, "\n")
            
            # Test the model on testing set
                if i % 100 == 0:
                    if self.args.load_constraints:
                        test_f1_score, test_acc_score = self.test_models()
                        test_f1_scores.append(test_f1_score)
                        test_acc_scores.append(test_acc_score)

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
        fig, axes = plt.subplots(2)
        axes[0].plot(critic_losses)
        axes[0].set_title("Critic Loss")
        axes[1].plot(model_bce_losses)
        axes[1].set_title("BCE Loss")

        # axes[2].plot(test_f1_scores)
        # axes[2].set_title("Test F1 Score")

        # axes[3].plot(test_acc_scores)
        # axes[3].set_title("Test Accuracy")


        plt.show()
        self.save_results(train_results)
        self.save()
    
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
    

    def load_balance_loss(self):
        # Max over actions
        #(torch.tensor(action_probs))
        # Keep count of actions per layer
        # When Batch is done:
        #   Compute action distribution for each layer
        #   PROBLEM: the distribution will not reflect what layers are actually possiblly
        pass

    def save(self):
        model_dict = {
            "router" : self._router.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
            "adv_avg" : self.adv_avg,
            "adv_std" : self.adv_std
        }
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

    def load_data(self):
        # Load training data
        with open(self.args.train_data, 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            print(fsz)
            self.train_data = np.load(f, allow_pickle=True)
            while f.tell() < fsz:
                self.train_data = np.vstack((self.train_data, np.load(f, allow_pickle=True)))
    
        np.random.shuffle(self.train_data)

        # Load testing data
        with open(self.args.test_data, 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            print(fsz)
            self.test_data = np.load(f, allow_pickle=True)
            while f.tell() < fsz:
                self.test_data = np.vstack((self.test_data, np.load(f, allow_pickle=True)))
    
        np.random.shuffle(self.test_data)


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
        # build model based on constraints only without router
        models = {}
        for k,v in self.args.constraints.items(): 
            failed, model = self._router.build_fixed(v)
            
            # Verify that the given constraints are valid
            assert not failed

            models[k] = model
        
        # Calculate the accuracy
        y_trues = np.zeros(len(self.test_data))
        y_preds = np.zeros(len(self.test_data))
        for i, ex in enumerate(self.test_data):
            task = ex[0]
            img = torch.tensor(ex[1]).cuda()
            
            # Get predicted binary classification
            pred_model = models[task](img)
            pred = torch.sigmoid(models[task](img))
            y_preds[i] = int(pred >= self.args.decision_threshold)
            y_trues[i]  = ex[2]

        test_f1_score = f1_score(y_trues, y_preds)
        print("F1 score of testing dataset", test_f1_score)
        test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
        print("Accuracy of testing dataset", test_acc_score)

        return test_f1_score, test_acc_score

    def save_results(self, train_results):
        with open(self.args.results, "w") as f:
            json.dump(train_results, f)