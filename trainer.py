import torch
import torch.optim as optim
from router import Router


class Trainer:
    def __init__(self, args):
        self.args = args
        self._router = Router(args, 1)
    
    def train(self):
        self.optimizer = optim.SGD(self._router.parameters(), lr=1e-3)
        for i in range(1000):
        
            failed, action_probs, actions, model= self._router(torch.rand(1, 1, 28, 28), torch.tensor([0]))
            self.optimizer.zero_grad()
            
            print("actions", actions)
            print(torch.tensor(actions) == 0)
            rewards = torch.ones_like(torch.tensor(action_probs))
            rewards = (torch.tensor(actions) != 0) * rewards
            if failed:
                rewards[-1] = -1
            
                
            returns = self.compute_return(rewards)
            print("returns", returns)
            loss = 0
            for t in range(len(action_probs)):
                loss += -action_probs[t] * returns[t]
            
            

            
            
            print("cur_loss", loss)
            loss.backward()
            self.optimizer.step()
            print(failed, model)

    
    def compute_return(self, rewards):
        returns = torch.zeros_like(rewards)
        for t in range(len(rewards)):
            if t + 1 > len(rewards):

                returns[t] = rewards[t] + self.args.gamma * returns[t+1]
            else:
                returns[t] = rewards[t]
        return returns
