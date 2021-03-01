import torch
import torch.nn as nn

import torch.optim as optim
import torchvision
from router import Router


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=16, shuffle=True)


for i, example in enumerate(train_loader):
    examples = example
    break

ce_loss = nn.CrossEntropyLoss()

class Trainer:
    def __init__(self, args):
        self.args = args
        self._router = Router(args, 1)
    
    def train(self):
        # TODO: Need to force last layer to have 10 outputs
        self.optimizer = optim.RMSprop(self._router.parameters(), lr=1e-4)
        total_loss = None
        for i in range(2000):
            failed, action_probs, actions, model= self._router(torch.rand(1, 1, 28, 28), torch.tensor([0]))
            
            
            rewards = torch.ones_like(torch.tensor(action_probs))
            rewards = (torch.tensor(actions) != 0) * rewards
            if failed:
                rewards[-1] = -10
            else:
                model_optim = optim.SGD(model.parameters(), lr=1e-3)
                print(model)
                print(examples[0].shape)
                pred = model(examples[0])
                loss = ce_loss(pred, examples[1])
                print("MODEL LOSS: ", loss)
                loss.backward()
                model_optim.step()
                model_optim.zero_grad()
                rewards[-1] += -loss.detach().numpy()
                
            returns = self.compute_return(rewards)
            print("returns", returns)
            loss = 0
            for t in range(len(action_probs)):
                loss += -action_probs[t] * returns[t]
            
            # Update router model
            loss.backward()
            if total_loss is None:
                total_loss = loss / self.args.batch_size
            else:
                total_loss += loss / self.args.batch_size

            if (i + 1) % self.args.batch_size == 0:
                self.optimizer.step()
                total_loss = None
                self.optimizer.zero_grad()

            
            print(failed, model)

    
    def compute_return(self, rewards):
        returns = torch.zeros_like(rewards)
        for t in range(len(rewards)):
            if t + 1 < len(rewards):

                returns[t] = rewards[t] + self.args.gamma * returns[t+1]
            else:
                returns[t] = rewards[t]
        return returns
