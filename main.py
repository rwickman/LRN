import argparse

# from configs.omniglot import mixed_config
from configs.omniglot import mixed_config

from trainer import Trainer

import torch
def main(args):
    #module_trainer = Trainer(args, mixed_config)
    module_trainer = Trainer(args, mixed_config)
    module_trainer.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general_args = parser.add_argument_group("General")
    general_args.add_argument("--load", action="store_true",
        help="Load prior progress.")
    general_args.add_argument("--save_dir", default="model",
        help="Directory to save models.")

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
        help="Dropout rate.")
    router_args.add_argument("--in_channels", type=int, default=1,
        help="Dropout rate.")

    hyper_args = parser.add_argument_group("HyperParameters")
    hyper_args.add_argument("--lr", type=float, default=2e-4,
        help="Router learning rate.")
    hyper_args.add_argument("--batch_size", type=int, default=32,
        help="Batch size.")
    hyper_args.add_argument("--epochs", type=int, default=100,
        help="Number of epochs to train the model.")

    hyper_args.add_argument("--min_temperature", type=float, default=1.0,
        help="Temperature to use for router.")    
    hyper_args.add_argument("--temperature", type=float, default=5.0,
        help="Temperature to use for router.")
    hyper_args.add_argument("--temp_decay_steps", type=int, default=100,
        help="Temperature to use for router.")
        
    hyper_args.add_argument("--entropy_lam", type=float, default=0.1,
        help="Entropy lambda.")        




    main(parser.parse_args())
