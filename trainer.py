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
        self.optimizer = optim.RMSprop(self._router.parameters(), lr=1e-5)#, betas=(0.5, 0.59))
        self.avg_return = torch.zeros(self.args.max_depth)
        self._layer_count = {}

    def train(self):
        # TODO: Need to force last layer to have 10 outputs
        total_loss = None
        for i in range(10000):
            #print(examples[0][i%self.args.batch_size:i%self.args.batch_size+1].shape)
            task = torch.tensor([train_out[i % len(train_out)][0]]).cuda()
            #plt.imshow(train_out[i % len(train_out)][1][0].transpose(1,2,0))
            #plt.show()
            img = torch.tensor(train_out[i % len(train_out)][1]).cuda()
            target = torch.tensor(train_out[i % len(train_out)][2], dtype=torch.float32).unsqueeze(0).cuda()
            #print("task", task, " target", target)

            router_result, added_layers = self._router(img, task)
            action_probs, actions, model = router_result.action_probs, router_result.actions, router_result.model 
            
            rewards = torch.zeros_like(torch.tensor(actions), dtype=torch.float64)
            #print("BEFORE", rewards)
            # if len(added_layers) > 0:
            #     rewards[:len(added_layers)] += self.layer_reward_bonus(added_layers)
            #print("AFTER", rewards)
            #rewards = (torch.tensor(actions) != 0) * 0.1
            #rewards = rewards + (torch.tensor(actions) == 0) * -0.1 
            if router_result.failed:
                rewards[-1] = -2
            else:
                if i > 5000:
                    action_task_pair = ((task.cpu().numpy()[0], target.cpu().numpy()[0]), tuple(actions)) 
                    if action_task_pair not in action_sets:
                        action_sets[action_task_pair] = 0
                    action_sets[action_task_pair] +=1
                    
                rewards[:len(added_layers)] += self.layer_reward_bonus(added_layers)
                model_optim = optim.SGD(model.parameters(), lr=1e-4)
                #print(model)
                pred = model(img)
                loss = bce_loss_fn(pred[0], target)
                #print("MODEL LOSS: ", loss)

                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                rewards[-1] += -loss.detach().cpu().numpy()
                # print("rewards", rewards)

            # act_tuple = tuple(actions)
            # if not failed and act_tuple not in action_sets:
            #     rewards[-1] = 10
            #     action_sets.add(act_tuple)
            

            returns = self.compute_return(rewards, action_probs)
            
            advantage = torch.zeros_like(returns) 
            for j in range(len(returns)):
                advantage[j] = (returns[j] - self.avg_return[j]) * 0.99 + returns[j] * (1-0.99) 
                if not router_result.failed:
                    self.avg_return[j] = (1-0.1)*self.avg_return[j] + 0.1*returns[j]
                

            #advantage = (advantage - advantage.mean()) / advantage.std()
            #print("returns", returns, "advantage", advantage, "avg_returns", self.avg_return)
            # if not failed:
            #     import sys
            #     sys.exit()
            
            loss = torch.zeros(1).cuda()
            for t in range(len(action_probs)):
                loss += -torch.log(action_probs[t][0][actions[t]]) * advantage[t]
                # if not router_result.failed and action_probs[t][0][actions[t]] > 0.25 and action_probs[t][0][actions[t]] != 0:
                #     loss += action_probs[t][0][actions[t]] * 10
                # Add entropy regularization
            
                loss += -self.args.entropy_lam * torch.sum(action_probs[t][0] * torch.log(action_probs[t][0] +1e-10))
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
                torch.nn.utils.clip_grad_norm_(self._router.parameters(), 1.0)
                self.optimizer.step()
                total_loss = None
                
                self.optimizer.zero_grad()

            print("action_probs[0]", action_probs[0])
            print(router_result.failed, model, actions)
        for pair in sorted(list(action_sets.items()), key=lambda a: a[1]):
            print(pair)

    
    def compute_return(self, rewards, action_probs):
        returns = torch.zeros_like(rewards).cuda()
        for t in reversed(range(len(rewards))):
            if t + 1 < len(rewards):
                returns[t] = rewards[t] + self.args.gamma * returns[t+1]
            else:
                returns[t] = rewards[t]

        return returns

    def layer_reward_bonus(self, added_layers):
        bonus = torch.zeros(len(added_layers))
        for i in range(len(added_layers)):
            if added_layers[i] not in self._layer_count:
                self._layer_count[added_layers[i]] = 1
            #bonus[i] += (1 / np.log(self._layer_count[added_layers[i]] + 1)) * 0.1
            bonus[i] += 1 / self._layer_count[added_layers[i]]
            self._layer_count[added_layers[i]] += 1
    
        return bonus