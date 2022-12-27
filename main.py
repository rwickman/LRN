import argparse
import os

# from configs.omniglot import mixed_config
from configs.omniglot import mixed_config, mixed_config_only_conv, mixed_config_only_conv_any_layer 


from taskrouter.trainer import Trainer
from taskrouter.benchmark import BenchmarkTrainer

import torch
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
        trainer = Trainer(args, config)

    if args.test:
        trainer.test()
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general_args = parser.add_argument_group("General")
    general_args.add_argument("--load", action="store_true",
        help="Load prior progress.")
    general_args.add_argument("--save_dir", default="model",
        help="Directory to save models.")
    general_args.add_argument("--benchmark", action="store_true",
        help="Train benchmark.")
    general_args.add_argument("--benchmark_model", default="3x3",
        help="The benchmark model to run.")
    
    general_args.add_argument("--test", action="store_true",
        help="Test model on testing set.")


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


    disc_args = parser.add_argument_group("Discriminator")
    disc_args.add_argument("--disc_enc_layers", type=int, default=6,
        help="Number of transformer encoder layers")
    disc_args.add_argument("--disc_dff", type=int, default=2048,
        help="DFF size for discriminator")
    disc_args.add_argument("--disc_num_heads", type=int, default=8,
        help="DFF size for discriminator")
    disc_args.add_argument("--disc_num_enc_layers", type=int, default=8,
        help="DFF size for discriminator")
    disc_args.add_argument("--disc_lr", type=float, default=1e-3,
        help="DFF size for discriminator")
    
    hyper_args = parser.add_argument_group("HyperParameters")
    hyper_args.add_argument("--lr", type=float, default=2e-4,
        help="Modules learning rate.")
    hyper_args.add_argument("--router_lr", type=float, default=2e-4,
        help="Router learning rate.")
    hyper_args.add_argument("--batch_size", type=int, default=32,
        help="Batch size.")
    hyper_args.add_argument("--epochs", type=int, default=100,
        help="Number of epochs to train the model.")

    hyper_args.add_argument("--min_temperature", type=float, default=1.0,
        help="Temperature to use for router.")    
    hyper_args.add_argument("--temperature", type=float, default=5.0,
        help="Temperature to use for router.")
    hyper_args.add_argument("--temp_decay_steps", type=int, default=10000,
        help="Temperature to use for router.")
    hyper_args.add_argument("--max_prob", type=float, default=0.5,
        help="Max probability for module.")
    hyper_args.add_argument("--min_prob", type=float, default=0.0,
        help="Min probability for module.")
    hyper_args.add_argument("--warm_up_steps", type=int, default=2080,
        help="Number of epochs for warmup .")
    hyper_args.add_argument("--warm_up_lr", type=float, default=1e-7,
        help="Number of epochs for warmup .")

    hyper_args.add_argument("--entropy_lam", type=float, default=0.1,
        help="Entropy lambda.")        
    hyper_args.add_argument("--cond_entropy_lam", type=float, default=0.1,
        help="Entropy lambda.")      



    main(parser.parse_args())
