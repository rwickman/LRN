import wandb
import random, os
from taskrouter.trainer import Trainer

class HyperparameterTuner:
    def __init__(self, args, config, train_dataset, val_dataset, test_dataset=None):
        self.args = args
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        self.models_path = f"models/{self.args.dataset}"
        self.project = f"{self.args.dataset}-lrn"
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def start(self):
        sweep_id = self.create_sweep_config()
        wandb.agent(sweep_id, function=self.hyperparameter_tune_main, count=8)

    def hyperparameter_tune_main(self):
        wandb.init(project=self.project, entity="lrn")
        
        self.args.save_dir = f"{self.models_path}/models_{len(os.listdir(self.models_path))}"
        self.args.lr = wandb.config.lr
        self.args.router_lr = wandb.config.router_lr
        self.args.cv_lam = wandb.config.cv_lam
        self.args.num_enc_layers = wandb.config.num_enc_layers
        self.args.min_temperature = wandb.config.min_temperature
        self.args.temp_decay_steps = int(wandb.config.temp_decay_steps)
        self.args.batch_size = wandb.config.batch_size
        trainer = Trainer(self.args, self.config, self.train_dataset, self.val_dataset, self.test_dataset)
        epochs = wandb.config.epochs
        
        for epoch in range(epochs):
            train_loss, bal_cv_loss, val_acc, val_loss = trainer.train()
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "bal_cv_loss": bal_cv_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            })

    def create_sweep_config(self):
        sweep_configuration = {
            'method': 'random',
            'name': 'sweep',
            'metric': {'goal': 'maximize', 'name': 'val_acc'},
            'parameters': 
            {
                'epochs': {'values': [75, 100, 125, 150, 175, 200]},
                'lr': {'max': 1e-3, 'min': 1e-5},
                'router_lr': {'max': 1e-3, 'min': 1e-5},
                'cv_lam': {'max': 2.0, 'min': 0.0},
                'num_enc_layers': {'values': [1, 2, 3, 4, 5, 6]},
                'min_temperature': {'max': 1.0, 'min': 0.001},
                'temp_decay_steps': {'max': 100000, 'min': 10000},
                'batch_size': {'values': [32, 64, 128]}
            }
        }

        sweep_id = wandb.sweep(sweep=sweep_configuration, project=self.project)

        return sweep_id
