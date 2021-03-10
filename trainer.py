import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from router import Router


with open("util/mnist-mtl.npy", 'rb') as f:
  fsz = os.fstat(f.fileno()).st_size
  print(fsz)
  train_out = np.load(f, allow_pickle=True)
  while f.tell() < fsz:
      train_out = np.vstack((train_out, np.load(f, allow_pickle=True)))
  
np.random.shuffle(train_out)
bce_loss_fn = nn.BCEWithLogitsLoss()


action_sets = {}

class Trainer:
    def __init__(self, args):
        self.args = args
        self._router = Router(args, 10).cuda()
        #print(self._router.parameters())
        self.optimizer = optim.RMSprop(self._router.parameters(), lr=1e-4)#, betas=(0.5, 0.59))
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
        for i in range(self.args.train_iter):
            #print(examples[0][i%self.args.batch_size:i%self.args.batch_size+1].shape)
            task = torch.tensor([train_out[i % len(train_out)][0]]).cuda()
            if train_out[i % len(train_out)][0] not in [0, 1]:
                continue
            #plt.imshow(train_out[i % len(train_out)][1][0].transpose(1,2,0))
            #plt.show()
            img = torch.tensor(train_out[i % len(train_out)][1]).cuda()
            target = torch.tensor(train_out[i % len(train_out)][2], dtype=torch.float32).unsqueeze(0).cuda()
            

            router_result = self._router(img, task)
            action_probs, actions, model, critic_outputs = router_result.action_probs, router_result.actions, router_result.model, router_result.critic_outputs  
            
            rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64).cuda()
            if len(actions) > 0:
                bonus_mask = torch.tensor(actions) != 0
                bonus_mask[-1] = not router_result.failed
                rewards[bonus_mask] += self.layer_reward_bonus(actions, bonus_mask)
            
            if router_result.failed:
                rewards[-1] = -5
            else:
                if i > 5000:
                    action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                    if action_task_pair not in action_sets:
                        action_sets[action_task_pair] = 0
                    action_sets[action_task_pair] +=1
                model_optim = optim.SGD(model.parameters(), lr=1e-3)
                pred = model(img)
                
                loss = bce_loss_fn(pred[0], target)
                model_bce_losses.append(loss.detach().cpu().numpy())
                print("MODEL LOSS: ", loss, "pred", torch.sigmoid(pred)[0].detach().cpu().numpy()[0], "vs", target.cpu().numpy()[0])

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

            for t in range(len(action_probs)):
                adv = returns[t] - critic_outputs[t]
                critic_loss = self.critic_loss_fn(critic_outputs[t].squeeze(0), tdlamret[t]) * (1/len(critic_outputs)) * self.args.critic_lam 

                loss += critic_loss
                total_critic_loss += critic_loss.detach().cpu().numpy()
                
                delta = advs[t] - self.adv_avg[t] 
                #adv = (delta) / self.adv_std[t]**0.5
                adv = (advs[t] - self.adv_avg[t]) / self.adv_std[t]**0.5
                adv = max(min(3.0, adv), -3.0)
                loss += -torch.log(action_probs[t][0][router_result.action_indices[t]]) * adv
                self.adv_std[t] = (1-self.args.adv_alpha) * (self.adv_std[t] + self.args.adv_alpha * delta**2) 
                self.adv_avg[t] = self.adv_avg[t] + self.args.adv_alpha * delta 
                advantage[t] = adv
    
                # if not router_result.failed and action_probs[t][0][actions[t]] > 0.50 and action_probs[t][0][actions[t]] != 0:
                #     loss += action_probs[t][0][actions[t]] * 10

                # Add entropy regularization
                loss += -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10))
            
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

            if (i + 1) % (self.args.batch_size * 2) == 0:
                #print(self._router.state_encoder.fc1.weight.grad)
                # print("self._router.state_encoder.fc1.weight.grad.max()", self._router.state_encoder.fc1.weight.grad.max())
                # print("self._router.critic_out.weight.grad.max()", self._router.critic_out.weight.grad.max())
                # print("self._router.task_embs.weight.grad.max()", self._router.task_embs.weight.grad.max())
                #print("self._router.att.state_W.weight.grad.max()", self._router.att.state_W.weight.grad.max())
                #print("self._router.att.state_W.weight.grad.max()", self._router.att.W_v.weight.grad.max())
                # print("self._router.task_embs.weight.grad", self._router.task_embs.weight.grad)

                #torch.nn.utils.clip_grad_norm_(self._router.parameters(), 1.0)
                #print("self._router.critic_out.weight.grad", self._router.critic_out.weight.grad)
                #print("self._router.state_encoder.fc1.weight.grad", self._router.state_encoder.fc1.weight.grad)
                #print("self._router.state_encoder.fc1.weight.grad", self._router.state_encoder.fc1.weight.grad.max())
                self.optimizer.step()
                total_loss = None
                
                self.optimizer.zero_grad()
            if i % 64 == 0:
                print("\ntask", task, " target", target)
                print("action_probs[0]", action_probs[0])
                if len(action_probs) > 1:
                    print("action_probs[1]", action_probs[1])
                if len(action_probs) > 2:
                    print("action_probs[2]", action_probs[2])
                print("Critic Outputs", torch.tensor(critic_outputs).cpu().numpy(), "VS", returns.cpu().numpy(), advantage.detach().cpu().numpy())
                # print("Avg ADV", self.adv_avg)
                # print("STD ADV", self.adv_std)

                print(router_result.failed, model, actions, "\n")
        for pair in sorted(list(action_sets.items()), key=lambda a: a[1]):
            print(pair)
        
        fig, axes = plt.subplots(2)
        axes[0].plot(critic_losses)
        axes[1].plot(model_bce_losses)
        plt.show()
        self.save()
    
    def compute_return(self, rewards, action_probs, critic_outputs):
        returns = torch.zeros_like(rewards, dtype=torch.float32).cuda()
        adv = torch.zeros_like(rewards, dtype=torch.float32).cuda()

        for t in reversed(range(len(rewards))):
            if t + 1 < len(rewards):
                returns[t] = rewards[t] + self.args.gamma * returns[t+1]
                
                # TD Residual
                delta = rewards[t] + self.args.gamma * critic_outputs[t+1].detach() -  critic_outputs[t].detach()
                
                # GAE
                adv[t] = delta + self.args.gamma * self.args.gae_lam * adv[t+1]
            else:
                returns[t] = rewards[t]
                adv[t] = returns[t] -  critic_outputs[t].detach()
        
        tdlamret = adv + torch.tensor(critic_outputs).cuda()
        
        return returns, adv, tdlamret

    def layer_reward_bonus(self, actions, bonus_mask):
        bonus = torch.zeros(torch.sum(bonus_mask)).cuda()
        b_i = 0
        for i in range(len(bonus)):
            if bool(bonus_mask[i]):
                if actions[i] not in self._layer_count:
                    self._layer_count[actions[i]] = 1
                bonus[b_i] += (1 / np.log(self._layer_count[actions[i]] + 1))
                
                #bonus[b_i] += 1 / self._layer_count[actions[i]]
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
