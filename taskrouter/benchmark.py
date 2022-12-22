import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, json

from taskrouter.helpers.data_loader import load_omniglot



class Conv3x3Layer(nn.Module):
    def __init__(self, stride, in_channels=48, out_channels=48):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1),
                nn.GroupNorm(4, out_channels), 
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride),
                nn.GroupNorm(4, out_channels), 
                nn.ReLU()
            )
    
    def forward(self, x):
        x = self.layer(x)
        return x

class Conv5x5Layer(nn.Module):
    def __init__(self, stride, in_channels=48, out_channels=48):
        super().__init__()
        self.layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=1),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 5, padding=2, stride=stride),
                    nn.GroupNorm(4, out_channels), 
                    nn.ReLU()
                )
    
    def forward(self, x):
        x = self.layer(x)
        return x



class Simple3x3Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.layers = []
        self.args = args
        self.config = config
        self._max_out_size = max(self.config.out_sizes)
        
        
        self.shared_inp = nn.Sequential(
            nn.Conv2d(1, 48, 1),
            nn.GroupNorm(4, 48),
            nn.ReLU()
        )
        
        

        for i in range(self.config.routing_depth):
            if self.args.benchmark_model == "3x3":
                self.layers.append(Conv3x3Layer(self.config.strides[i]))
            else:
                self.layers.append(Conv5x5Layer(self.config.strides[i]))
                
        
        self.layers = nn.Sequential(*self.layers)
        out_layers = []
        out_in_size = 48 * 4**2
        for i in range(len(self.config.out_sizes)):
            out_layers.append(
                nn.Linear(out_in_size, self.config.out_sizes[i])
            )
        self._out_layers = nn.ModuleList(out_layers)

        


    def forward(self, x, task_ids):
        batch_size = x.shape[0]
        y = torch.zeros(batch_size, self._max_out_size, device=self.config.device)
        x = self.shared_inp(x)
        x = self.layers(x)
        x = x.view(batch_size, -1)

        for i in range(batch_size):
            y[i, :self.config.out_sizes[task_ids[i, 0]]] = self._out_layers[task_ids[i]](x[i])
        
        return y

            
        # # for task_id in task_ids:
        #     self.lay
# from configs.omniglot import mixed_config
# mixed_config.out_sizes = [3, 3]
# m = Simple3x3Model(mixed_config).to(mixed_config.device)
# fake_x = torch.randn(2, 1, 105, 105, device=mixed_config.device)
# task_ids = torch.tensor([0, 1], device=mixed_config.device).unsqueeze(1)
# print("fake_x.shape", fake_x.shape)
# print("task_ids.shape", task_ids.shape)
# out = m(fake_x, task_ids)
# print(out)

class BenchmarkTrainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        train_dataset, val_dataset, test_dataset, out_sizes = load_omniglot()
        self.config.out_sizes = out_sizes

        self.model = Simple3x3Model(self.args, self.config).to(self.config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model_file = os.path.join(self.args.save_dir, "model.pt")
        self.train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_dataloader = DataLoader(
            train_dataset,
            self.args.batch_size,
            num_workers=4,
            prefetch_factor=8,
            drop_last = True,
            shuffle=True,
            multiprocessing_context="spawn")

        self.val_dataloader = DataLoader(
            val_dataset,
            self.args.batch_size,
            num_workers=4,
            prefetch_factor=8,
            drop_last = False,
            shuffle=True,
            multiprocessing_context="spawn")
        
        self.test_dataloader = DataLoader(
            test_dataset,
            self.args.batch_size,
            num_workers=4,
            prefetch_factor=8,
            drop_last = False,
            shuffle=True,
            multiprocessing_context="spawn")

        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }
        if self.args.load:
            self.load()

    def test(self):
        print("RUNNING VALIDATION")
        total_val_loss = 0
        num_exs = 0
        self.model.eval()
        num_correct = 0
        for imgs, task_ids, class_ids in tqdm(self.test_dataloader):
            with torch.no_grad():
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)
                y_preds = self.model(imgs, task_ids)

                
                loss = self.loss_fn(y_preds, class_ids)
                
                num_correct += (y_preds.argmax(axis=1) == class_ids).sum().item()
                total_val_loss += (loss.item() * imgs.shape[0])

                num_exs += imgs.shape[0]

            # if num_exs > 256:
            #     break
        val_acc = num_correct / num_exs

        self.model.train()
        print("TEST ACCURACY", val_acc)

    def save(self):
        print("SAVING MODEL")
        model_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        torch.save(model_dict, self.model_file)
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)

    def load(self):
        print("LOADING MODEL")
        model_dict = torch.load(self.model_file)
        self.model.load_state_dict(model_dict["model"])

        self.optimizer.load_state_dict(model_dict["optimizer"])

        with open(self.train_dict_file) as f:
            train_dict = json.load(f)
            for key in train_dict:
                self.train_dict[key] = train_dict[key]
        
        self.optimizer.param_groups[0]["lr"] = self.args.lr


    def validate(self):
        print("RUNNING VALIDATION")
        total_val_loss = 0
        num_exs = 0
        self.model.eval()
        num_correct = 0
        for imgs, task_ids, class_ids in tqdm(self.val_dataloader):
            with torch.no_grad():
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)
                y_preds = self.model(imgs, task_ids)

                
                loss = self.loss_fn(y_preds, class_ids)
                
                num_correct += (y_preds.argmax(axis=1) == class_ids).sum().item()
                total_val_loss += (loss.item() * imgs.shape[0])

                num_exs += imgs.shape[0]

            # if num_exs > 256:
            #     break
        val_acc = num_correct / num_exs
        print("VALIDATION ACCURACY", val_acc)
        self.train_dict["val_loss"].append(total_val_loss/num_exs)
        self.train_dict["val_acc"].append(val_acc)
        self.model.train()

    def train(self):
        for e_i in range(self.args.epochs):
            total_train_loss = 0
            num_exs = 0

            for imgs, task_ids, class_ids in tqdm(self.train_dataloader):
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)

                self.optimizer.zero_grad()
                # self.optimizer.zero_grad()
                #print("self.args.temperature", self.args.temperature)
                y_preds = self.model(imgs, task_ids)
                # print("logits", logits)

                loss = self.loss_fn(y_preds, class_ids)

                loss.backward()
                total_train_loss += (loss.item() * imgs.shape[0])
                num_exs += imgs.shape[0]
                # print(self.lrn_model.router_model.trans_router.actor_out2.weight.grad.min(), self.lrn_model.router_model.trans_router.actor_out2.weight.grad.max())

                # import matplotlib.pyplot as plt
                # plt.imshow(imgs[0, 0].detach().cpu())
                # plt.show()
                self.optimizer.step()
                #self.optimizer.step()
            
            self.train_dict["train_loss"].append(total_train_loss/num_exs)
            self.validate()
            self.save()