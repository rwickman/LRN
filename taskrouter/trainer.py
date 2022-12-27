import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, json
import numpy as np
import matplotlib.pyplot as plt



from taskrouter.helpers.data_loader import load_omniglot
from taskrouter.models.discriminator import Discriminator
from taskrouter.models.lrn_model import LRNModel
from taskrouter.helpers.model_util import create_disc_train_data, compute_entropy_loss
class Trainer:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
        train_dataset, val_dataset, test_dataset, out_sizes = load_omniglot()
        
        self.config.out_sizes = out_sizes
        
        # Create the models
        self.lrn_model = LRNModel(self.args, self.config).to(self.config.device) 
        self.disc = Discriminator(self.args, self.config).to(self.config.device)

        # Create the optimizers
        self.optimizer_router = optim.Adam(self.lrn_model.router_model.parameters(), lr=self.args.router_lr)
        self.optimizer_modular = optim.Adam(self.lrn_model.modular_model.parameters(), lr=self.args.lr)
        self.optimizer_disc = optim.Adam(self.disc.parameters(), lr=self.args.disc_lr)
        # self.optimizer = optim.Adam(self.lrn_model.parameters(), lr=self.args.lr)

        # print("out_sizes", out_sizes)
        self.model_file = os.path.join(self.args.save_dir, "model.pt")
        self.train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")
        

        # Create the datasets
        self.train_dataloader = DataLoader(
            train_dataset,
            self.args.batch_size,
            num_workers=2,
            prefetch_factor=8,
            drop_last = True,
            shuffle=True)

        self.val_dataloader = DataLoader(
            val_dataset,
            self.args.batch_size,
            num_workers=4,
            prefetch_factor=8,
            drop_last = False,
            shuffle=True)
        
        self.test_dataloader = DataLoader(
            test_dataset,
            self.args.batch_size,
            num_workers=4,
            prefetch_factor=8,
            drop_last = False,
            shuffle=True)


        self.train_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "train_modules": [],
            "val_modules": [],
            "num_steps": 0
        }

        self.loss_fn = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        if self.args.load:
            self.load()

    def adjust_lr(self):
        if self.train_dict["num_steps"] < self.args.warm_up_steps:
            inter_step = self.train_dict["num_steps"] / self.args.warm_up_steps
            print("LR", (1-inter_step) * self.args.warm_up_lr + inter_step * self.args.router_lr)
            self.optimizer_router.param_groups[0]["lr"] = (1-inter_step) * self.args.warm_up_lr + inter_step * self.args.router_lr
        else:
            self.optimizer_router.param_groups[0]["lr"] = lr


    def save(self):
        print("SAVING MODEL")
        model_dict = {
            "model": self.lrn_model.state_dict(),
            "disc": self.disc.state_dict(),
            "optimizer_router": self.optimizer_router.state_dict(),
            "optimizer_modular": self.optimizer_modular.state_dict(),
            "optimizer_disc": self.optimizer_disc.state_dict()
        }
        # model_dict = {
        #     "model": self.lrn_model.state_dict(),
        #     "optimizer": self.optimizer.state_dict()
        # }

        torch.save(model_dict, self.model_file)
        with open(self.train_dict_file, "w") as f:
            json.dump(self.train_dict, f)

    def load(self):
        print("LOADING MODEL")
        model_dict = torch.load(self.model_file)
        self.lrn_model.load_state_dict(model_dict["model"])
        self.disc.load_state_dict(model_dict["disc"])
        # self.optimizer.load_state_dict(model_dict["optimizer"])
        self.optimizer_router.load_state_dict(model_dict["optimizer_router"])
        self.optimizer_modular.load_state_dict(model_dict["optimizer_modular"])
        self.optimizer_disc.load_state_dict((model_dict["optimizer_disc"]))

        with open(self.train_dict_file) as f:
            train_dict = json.load(f)
            for key in train_dict:
                self.train_dict[key] = train_dict[key]
        
        self.optimizer_modular.param_groups[0]["lr"] = self.args.lr
        self.optimizer_router.param_groups[0]["lr"] = self.args.router_lr

        # print("self._optim.param_groups", self.optimizer.param_groups)
        # self.optimizer.param_groups[0]["lr"] = self.args.lr

    def test(self):
        print("RUNNING VALIDATION")
        total_val_loss = 0
        num_exs = 0
        modules = torch.zeros(self.config.num_modules)
        self.lrn_model.eval()
        num_correct = 0
        for imgs, task_ids, class_ids in tqdm(self.test_dataloader):
            with torch.no_grad():
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)
                y_preds, router_outs, _, _ = self.lrn_model(imgs, task_ids, argmax=True)
                
                loss = self.loss_fn(y_preds, class_ids)
                                
                num_correct += (y_preds.argmax(axis=1) == class_ids).sum().item()
                total_val_loss += (loss.item() * imgs.shape[0])

                num_exs += imgs.shape[0]
                modules += router_outs.detach().cpu().sum(axis=[0,1])
            # if num_exs > 256:
            #     break
        modules = modules/num_exs
        val_acc = num_correct / num_exs
        print("modules", modules)
        print("TEST ACCURACY", val_acc)

        self.lrn_model.train()


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
                y_preds, router_outs, _, _ = self.lrn_model(imgs, task_ids, argmax=True)
                
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
        r = -(np.log(self.args.min_temperature)- np.log(self.args.temperature)) / (self.args.temp_decay_steps)
        return max(self.args.temperature * np.e ** (-r * self.train_dict["num_steps"]), self.args.min_temperature)
        #return max((1 - train_epoch/self.args.temp_decay_steps) * (self.args.temperature - self.args.min_temperature) + self.args.min_temperature , self.args.min_temperature)

        
    def train(self):
        for e_i in range(self.args.epochs):
            total_train_loss = 0
            num_exs = 0
            modules = torch.zeros(self.config.num_modules)


            for imgs, task_ids, class_ids in tqdm(self.train_dataloader):
                self.adjust_lr()
                imgs, task_ids, class_ids = imgs.to(self.config.device), task_ids.to(self.config.device), class_ids.to(self.config.device)
                self.optimizer_router.zero_grad()
                self.optimizer_modular.zero_grad()


                #print("self.args.temperature", self.args.temperature)
                y_preds, router_outs, logits, action_masks = self.lrn_model(
                    imgs, task_ids, temperature=self.temperature)

                disc_preds = self.disc.predict(task_ids, logits.softmax(dim=2)).detach()[:, :-1] #* (1-action_masks)
                temp_logits = logits.view(-1, logits.shape[2])
                disc_preds = disc_preds.argmax(dim=2)

                router_entropy_loss = self.loss_fn(
                    temp_logits,
                    disc_preds.view(-1))


                # disc_seq, disc_seq_lens, disc_tgts, disc_data_idxs, disc_imgs = create_disc_train_data(imgs, logits.detach().softmax(dim=2), self.config)
                # disc_preds = self.disc(disc_imgs, disc_seq, disc_seq_lens, disc_data_idxs) #* (1-action_masks)
                # entropy_loss = self.loss_fn(
                #     logits.view(-1, logits.shape[2]),
                #     disc_preds.softmax(dim=1))



                loss = self.loss_fn(y_preds, class_ids)
                param_entropy_loss = compute_entropy_loss(self.config, logits)
                total_loss = loss - self.args.cond_entropy_lam * torch.clamp_max(router_entropy_loss, 100)  + self.args.entropy_lam * param_entropy_loss 
                total_loss.backward()
                self.optimizer_modular.step()
                self.optimizer_router.step()

                self.optimizer_disc.zero_grad()
                old_disc_tgts = logits.detach().clone()
                disc_seq, disc_seq_lens, disc_tgts, disc_data_idxs, disc_task_ids = create_disc_train_data(task_ids, logits.detach().softmax(dim=2), self.config)
                disc_preds = self.disc(disc_task_ids, disc_seq, disc_seq_lens, disc_data_idxs)

                
                # print("DISC SHAPES", disc_preds.shape, disc_tgts.shape)
                #print("disc_preds[:, 0:1]", disc_preds[:, 0:1])
                # print("disc_preds.shape", disc_preds.shape, "disc_tgts.shape", disc_tgts.shape)

                #print(" disc_tgts.softmax(dim=2).view(-1, disc_tgts.shape[2])",  disc_tgts.softmax(dim=2).view(-1, disc_tgts.shape[2]).shape)
               # print("disc_preds[:, 0:1].shape", disc_preds.shape)
                # print("disc_preds.shape", disc_preds.shape, "disc_tgts.shape", disc_tgts.shape)
                entropy_loss = self.loss_fn(
                    disc_preds,
                    disc_tgts.argmax(dim=1))
                # print("entropy_loss", entropy_loss)
                # print("self.temperature", self.temperature)
                entropy_loss.backward()
                self.optimizer_disc.step()


                # print(self.lrn_model.router_model.trans_router.actor_out2.weight.grad.min(), self.lrn_model.router_model.trans_router.actor_out2.weight.grad.max())

                # import matplotlib.pyplot as plt
                # plt.imshow(imgs[0, 0].detach().cpu())
                # plt.show()

                #self.optimizer.step()

                total_train_loss += (loss.item() * imgs.shape[0])
                num_exs += imgs.shape[0]
                modules += router_outs.detach().cpu().sum(axis=[0,1])
                self.train_dict["num_steps"] += 1
                if num_exs % 1024 == 0:
                    print("disc_preds", disc_preds[0:2])
                    print("disc_preds",disc_preds.argmax(dim=1).view(self.args.batch_size, 8))
                    print("disc_tgts", disc_tgts.argmax(dim=1).view(self.args.batch_size, 8))
                    print("old_disc_tgts", old_disc_tgts.argmax(dim=2).view(self.args.batch_size, 8))
                
                    print("modules", modules.tolist())
                    #print(self.lrn_model.router_model.task_embs(torch.tensor([1], device=self.config.device)))
                    print("task_embs grad", self.lrn_model.router_model.task_embs.weight.grad.min().item(), self.lrn_model.router_model.task_embs.weight.grad.max().item())
                    print("actor_out1 grad", self.lrn_model.router_model.trans_router.actor_out1.weight.grad.min().item(), self.lrn_model.router_model.trans_router.actor_out1.weight.grad.max().item())
                    if self.args.vit:
                        print("state_encoder grad", self.lrn_model.router_model.state_encoder.to_patch_embedding[-1].weight.grad.min().item(), self.lrn_model.router_model.state_encoder.to_patch_embedding[-1].weight.grad.max().item(), self.lrn_model.router_model.state_encoder.to_patch_embedding[-1].weight.grad.mean().item())
                    else:
                        print("state_encoder grad", self.lrn_model.router_model.state_encoder.model.features[0][0].weight.grad.min().item(), self.lrn_model.router_model.state_encoder.model.features[0][0].weight.grad.max().item())
                    print("self.lrn_model.modular_model out_layers grad", self.lrn_model.modular_model._out_layers[task_ids[0].unsqueeze(0)].weight.grad.min(), self.lrn_model.modular_model._out_layers[task_ids[0].unsqueeze(0)].weight.grad.max())
                    print("LRN loss", loss)
                    print("entropy_loss", entropy_loss)
                    print("router_entropy_loss", router_entropy_loss)
                    print("param_entropy_loss", param_entropy_loss)
                    print("self.temperature", self.temperature)
                    
                # plt.imshow(imgs[0, 0].detach().cpu())
                # plt.show()


            modules = modules/num_exs
            
            print("modules", modules)
            print("self.temperature", self.temperature)
            print("loss", loss)
            print("y_preds", y_preds[0:2].softmax(-1))
            print("class_ids", class_ids[0:2])
            self.train_dict["train_loss"].append(total_train_loss/num_exs)
            self.train_dict["train_modules"].append(modules.tolist())
            self.validate()
            self.save()
