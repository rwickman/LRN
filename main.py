import argparse

from trainer import Trainer

def main(args):
    model_trainer = Trainer(args) 
    model_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
        help="Load prior progress.")
    parser.add_argument("--save_dir", default="models",
        help="Directory to save models.")
    parser.add_argument("--img_size", type=int, default=28,
        help="Image square dimension size.")
    parser.add_argument("--in_channels", type=int, default=1,
        help="Number of input channels (RBG = 3, grayscale = 1).")
    parser.add_argument("--state_size", type=int, default=256,
        help="Latent size of the encoded state.")
    parser.add_argument("--layer_emb_size", type=int, default=256,
        help="Latent size of the layer embeddings.")
    parser.add_argument("--task_emb_size", type=int, default=256,
        help="Latent size of the task embeddings.")
    parser.add_argument("--hidden_size", type=int, default=256,
        help="Latent size of the hidden state.")
    parser.add_argument("--max_depth", type=int, default=4,
        help="Latent size of the hidden state.")
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size.")
    parser.add_argument("--drop_rate", type=float, default=0.2,
        help="Dropout rate for state encoder .")
    parser.add_argument("--num_dup_layers", type=int, default=2,
        help="Number of duplicate layers .")
    parser.add_argument("--num_classes", type=int, default=1,
        help="Number classes of output.")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Number classes of output.")
    parser.add_argument("--entropy_lam", type=float, default=0.001,
        help="Number classes of output.")
    main(parser.parse_args())