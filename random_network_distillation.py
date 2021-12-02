import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from configs import ale_config

class RandomNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.in_channels, self.args.hidden_size)
        self.fc2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.fc3 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.fc4 = nn.Linear(self.args.hidden_size, self.args.hidden_size*2)
        self.fc5 = nn.Linear(self.args.hidden_size*2, self.args.hidden_size*2)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        #self.act3 = nn.GELU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act2(self.fc3(x))
        x = self.act2(self.fc4(x))
        x = self.fc5(x)
        return x

def rand_loss(pred_state, rand_state):
    return ((pred_state - rand_state) ** 2).mean(axis=-1)


class Distillation:
    def __init__(self, args):
        self.args = args
        self.rnd_net = RandomNet(self.args).eval().to(ale_config.device)

        self.pred_net = RandomNet(self.args).to(ale_config.device)
        self.pred_optim = optim.Adam(self.pred_net.parameters(), lr=self.args.lr, eps=1e-4)
        if self.args.load:
            self.load()
    
    def get_intrinsic_reward(self, states):
        return rand_loss(self.pred_net(states), self.rnd_net(states))

    def train(self, states):
        losses = []
        for e_i in range(self.args.noptepochs):
            batch_ids = torch.randperm(len(states))
            for m_i in range(self.args.nminibatches):
                start = m_i * (len(states) // self.args.nminibatches)
                end = (m_i + 1) * (len(states) // self.args.nminibatches)
                m_batch_idxs = batch_ids[start:end]
                pred = self.pred_net(states[m_batch_idxs])
                rnd_pred = self.rnd_net(states[m_batch_idxs])
                loss = rand_loss(pred, rnd_pred).mean()
                self.pred_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pred_net.parameters(), self.args.max_grad_norm)
                self.pred_optim.step()
                losses.append(loss.item())
        #print("DISTILL LOSSES", losses)
        print("states", states[:2])
        print("\npred", pred[0:3])
        print("rnd_pred", rnd_pred[:3], "\n")
        return losses
    
    def save(self):
        model_dict = {
            "rnd_net" : self.rnd_net,
            "pred_net": self.pred_net,
            "pred_optim": self.pred_optim
        }

        torch.save(model_dict, os.path.join(self.args.save_dir, "pred_net.pt"))
    
    def load(self):
        model_dict = torch.load(os.path.join(self.args.save_dir, "pred_net.pt"))
        self.rnd_net = model_dict["rnd_net"]
        self.pred_net = model_dict["pred_net"]
        self.pred_optim = model_dict["pred_optim"]
