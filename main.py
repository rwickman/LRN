import argparse
import os

# from configs.omniglot import mixed_config
from configs.omniglot import mixed_config, mixed_config_only_conv, mixed_config_only_conv_any_layer 
from taskrouter.helpers.data_loader import load_omniglot, load_cifar
from configs.cifar import cifar_config
from taskrouter.trainer import Trainer
from taskrouter.benchmark import BenchmarkTrainer

import torch
from wandb_sweep import HyperparameterTuner


def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    #module_trainer = Trainer(args, mixed_config)
    if args.any_layer:
        config = mixed_config_only_conv_any_layer
    elif args.only_conv:
        config = mixed_config_only_conv
    else:
        config = mixed_config

    if args.benchmark:
        trainer = BenchmarkTrainer(args, mixed_config)        
    else:
        if args.dataset == "omniglot":
            train_dataset, val_dataset, test_dataset, out_sizes = load_omniglot()
            config.out_sizes = out_sizes
            trainer = Trainer(args, config, train_dataset, val_dataset, test_dataset)
        elif args.dataset == "cifar":
            config = cifar_config
            train_dataset, val_dataset = load_cifar()
            test_dataset = None 
            if not args.hyperparameter_tune:
                trainer = Trainer(args, config, train_dataset, val_dataset)

    if args.test:
        trainer.test()
    else:
        if args.hyperparameter_tune:
            # Perform hyperparameter tuning
            tuner = HyperparameterTuner(args, config, train_dataset, val_dataset, test_dataset)
            tuner.start()
        else:
            trainer.setup()
            for epoch in range(args.epochs):
                train_loss, bal_cv_loss, val_acc, val_loss = trainer.train()

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general_args = parser.add_argument_group("General")
    general_args.add_argument("--load", action="store_true",
        help="Load prior progress.")
    general_args.add_argument("--save_dir", default="model",
        help="Directory to save models.")
    general_args.add_argument("--dataset", default="omniglot",
        help="Dataset to train on.")

    general_args.add_argument("--benchmark", action="store_true",
        help="Train benchmark.")
    general_args.add_argument("--benchmark_model", default="3x3",
        help="The benchmark model to run.")
    general_args.add_argument("--test", action="store_true",
        help="Test model on testing set.")
    general_args.add_argument("--hyperparameter_tune", action="store_true",
        help="Use W&B to perform hyperparameter tuning.")
    general_args.add_argument("--debug", action="store_true",
        help="Print out debug info while training.")

    router_args = parser.add_argument_group("Router Parameters")
    router_args.add_argument("--emb_size", type=int, default=512,
        help="Embedding size.")
    router_args.add_argument("--dff", type=int, default=2048,
        help="Embedding size.")
    router_args.add_argument("--num_enc_layers", type=int, default=6,
        help="Number transformer encoder layers.")
    router_args.add_argument("--num_heads", type=int, default=8,
        help="Number of heads in transformer encoder.")
    router_args.add_argument("--drop_rate", type=float, default=0.1,
        help="Dropout rate.")
    router_args.add_argument("--patch_size", type=int, default=15,
        help="The patch size of the VIT.")
    router_args.add_argument("--in_channels", type=int, default=1,
        help="Number of input channels on the images.")
    router_args.add_argument("--vit", action="store_true",
        help="Use the VIT for the image processing.")
    router_args.add_argument("--only_conv", action="store_true",
        help="Only use convolutional layers for modules.")
    router_args.add_argument("--only_3x3", action="store_true",
        help="Only use convolutional layers for modules.")
    router_args.add_argument("--any_layer", action="store_true",
        help="Modules can be used at any layer where applicable.")
    router_args.add_argument("--decoder", action="store_true",
        help="Apply transformer decoder mask.")
    
    hyper_args = parser.add_argument_group("HyperParameters")
    hyper_args.add_argument("--lr", type=float, default=2e-4,
        help="Modules learning rate.")
    hyper_args.add_argument("--router_lr", type=float, default=2e-4,
        help="Router learning rate.")
    hyper_args.add_argument("--batch_size", type=int, default=64,
        help="Batch size.")
    hyper_args.add_argument("--epochs", type=int, default=100,
        help="Number of epochs to train the model.")

    hyper_args.add_argument("--min_temperature", type=float, default=0.5,
        help="Temperature to use for router.")    
    hyper_args.add_argument("--temperature", type=float, default=5.0,
        help="Temperature to use for router.")
    hyper_args.add_argument("--temp_decay_steps", type=int, default=10000,
        help="Number of steps before temperature decays to min_temperature.")
    hyper_args.add_argument("--max_prob", type=float, default=0.5,
        help="Max probability for module.")
    hyper_args.add_argument("--min_prob", type=float, default=0.0,
        help="Min probability for module.")
    hyper_args.add_argument("--warm_up_steps", type=int, default=0,
        help="Number of epochs for warmup .")
    hyper_args.add_argument("--warm_up_lr", type=float, default=1e-7,
        help="Number of epochs for warmup .")
    hyper_args.add_argument("--weight_decay", type=float, default=0.003,
        help="Number of epochs for warmup .")
    hyper_args.add_argument("--entropy_lam", type=float, default=0.01,
        help="Entropy lambda.")  
    hyper_args.add_argument("--cv_lam", type=float, default=0.1,
        help="Entropy lambda.")      
    hyper_args.add_argument("--lr_decay", type=float, default=0.999,
        help="Decay value for learning rate.")      


    main(parser.parse_args())
