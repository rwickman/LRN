import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lrn_model import LRNModel
import os, json
from model_util import balance_loss

from data_loader import load_omniglot

class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
        train_dataset, val_dataset, test_dataset, out_sizes = load_omniglot()
        
        self.config.out_sizes = out_sizes
        self.lrn_model = LRNModel(self.args, self.config).to(self.config.device) 
        self.optimizer = optim.Adam(self.lrn_model.parameters(), lr=self.args.lr)

        # print("out_sizes", out_sizes)
        self.model_file = os.path.join(self.args.save_dir, "model.pt")
        self.train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        

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

        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "train_modules": [],
            "val_modules": [],

        }
        self.loss_fn = nn.CrossEntropyLoss()

        if self.args.load:
            self.load()

    def save(self):
        print("SAVING MODEL")
        model_dict = {
            "model": self.lrn_model.state_dict(),
            "optimizer": self.optimizer.state_dict() 
        }

        torch.save(model_dict, self.model_file)
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)

    def load(self):
        print("LOADING MODEL")
        model_dict = torch.load(self.model_file)
        self.lrn_model.load_state_dict(model_dict["model"])
        self.optimizer.load_state_dict(model_dict["optimizer"])
        
        with open(self.train_dict_file) as f:
            train_dict = json.load(f)
            for key in train_dict:
                self.train_dict[key] = train_dict[key]
        # print("self._optim.param_groups", self.optimizer.param_groups)
        self.optimizer.param_groups[0]["lr"] = self.args.lr

    def validate(self):
        print("RUNNING VALIDATION")
        total_val_loss = 0
        num_exs = 0
        modules = torch.zeros(self.config.num_modules)
        self.lrn_model.eval()
        num_correct = 0
        for imgs, task_ids, class_ids in tqdm(self.val_dataloader):
            with torch.no_grad():
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)
                y_preds, router_outs, _ = self.lrn_model(imgs, task_ids, argmax=True)
                
                loss = self.loss_fn(y_preds, class_ids)
                
                num_correct += (y_preds.argmax(axis=1) == class_ids).sum().item()
                total_val_loss += (loss.item() * imgs.shape[0])

                num_exs += imgs.shape[0]
                modules += router_outs.detach().cpu().sum(axis=[0,1])
            # if num_exs > 256:
            #     break
        modules = modules/num_exs
        val_acc = num_correct / num_exs
        print("VALIDATION ACCURACY", val_acc)
        self.train_dict["val_loss"].append(total_val_loss/num_exs)
        self.train_dict["val_modules"].append(modules.tolist())
        self.train_dict["val_acc"].append(val_acc)
        self.lrn_model.train()
    
    @property
    def temperature(self):
        train_epoch = len(self.train_dict["train_loss"])

        return max((1 - train_epoch/self.args.temp_decay_steps) * (self.args.temperature - self.args.min_temperature) + self.args.min_temperature , self.args.min_temperature)

        
    def train(self):
        for e_i in range(self.args.epochs):
            total_train_loss = 0
            num_exs = 0
            modules = torch.zeros(self.config.num_modules)


            for imgs, task_ids, class_ids in tqdm(self.train_dataloader):
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)

                self.optimizer.zero_grad()
                #print("self.args.temperature", self.args.temperature)
                y_preds, router_outs, logits = self.lrn_model(imgs, task_ids, temperature=self.temperature)
                # print("logits", logits)

                loss = self.loss_fn(y_preds, class_ids)

                bal_loss = balance_loss(logits, self.config)

                
                total_loss = loss + self.args.entropy_lam * bal_loss


                total_loss.backward()
                
                self.optimizer.step()
                
                total_train_loss += (loss.item() * imgs.shape[0])
                num_exs += imgs.shape[0]
                modules += router_outs.detach().cpu().sum(axis=[0,1])
                if num_exs % 1024 == 0:
                    print(modules)

            modules = modules/num_exs
            print("modules", modules)
            print("self.temperature", self.temperature)
            self.train_dict["train_loss"].append(total_train_loss/num_exs)
            self.train_dict["train_modules"].append(modules.tolist())
            self.validate()
            self.save()
