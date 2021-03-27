import argparse
from classifier_trainer import ClassifierTrainer
from rl_trainer import RLTrainer
import os, sys
sys.path.append(os.path.abspath("benchmarks"))
from mnist_benchmarks import run_mnist_benchmarks

def main(args):
    if args.run_mnist_benchmarks:
        run_mnist_benchmarks(args)
    else:
        model_trainer = ClassifierTrainer(args) 
        model_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
        help="Load prior progress.")
    parser.add_argument("--num_tasks", type=int, default=10,
        help="Number of tasks.")
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
    parser.add_argument("--dff", type=int, default=512,
        help="Latent size of the hidden state.")
    parser.add_argument("--hidden_size", type=int, default=256,
        help="Latent size of the hidden state.")
    parser.add_argument("--max_depth", type=int, default=3,
        help="Latent size of the hidden state.")
    parser.add_argument("--lr", type=float, default=1e-4,
        help="Router learning rate.")
    parser.add_argument("--fb_lr", type=float, default=1e-4,
        help="Output model learning rate.")
    parser.add_argument("--batch_size", type=int, default=32,
        help="Batch size.")
    parser.add_argument("--drop_rate", type=float, default=0.1,
        help="Dropout rate.")
    parser.add_argument("--fb_drop_rate", type=float, default=0.2,
        help="Layer dropout rate for router.")
    parser.add_argument("--num_dup_layers", type=int, default=4,
        help="Number of duplicate layers .")
    parser.add_argument("--num_classes", type=int, default=1,
        help="Number classes of output.")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="Reward decay.")
    parser.add_argument("--gae_lam", type=float, default=0.95,
        help="GAE lambda.")
    parser.add_argument("--entropy_lam", type=float, default=0.001,
        help="Entropy lambda.")
    parser.add_argument("--critic_lam", type=float, default=0.5,
        help="Critic lambda.")
    parser.add_argument("--load_bal_lam", type=float, default=0.1,
        help="Load-balancing lambda.")
    parser.add_argument("--train_iter", type=int, default=10000,
        help="Number of training router iterations.")
    parser.add_argument("--epochs", type=int, default=100,
        help="Number of training epochs.")
    parser.add_argument("--adv_alpha", type=float, default=0.05,
        help="Number of training router iterations.")
    parser.add_argument("--num_heads", type=int, default=4,
        help="Number attention heads.")
    parser.add_argument("--num_enc_layers", type=int, default=1,
        help="Number of encoder layers used for router.")
    parser.add_argument("--max_pos_enc", type=int, default=10000,
        help="Maximum positional encoding for encoder.")
    parser.add_argument("--load_constraints", default=None,
        help="JSON constraints.")
    parser.add_argument("--train_data", default="data/mnist-mtl.npy",
        help="NPY train dataset.")
    parser.add_argument("--test_data", default="data/mnist-mtl-test.npy",
        help="NPY test dataset.")
    parser.add_argument("--decision_threshold", type=float, default=0.5,
        help="Decision threshold used for task classification.")
    parser.add_argument("--no_reward_bonus", action="store_true",
        help="Don't use a reward bonus.")
    parser.add_argument("--results", default="results.json",
        help="File to save JSON results.")
    parser.add_argument("--test_results", default="results.json",
        help="File to save JSON test results.")
    parser.add_argument("--run_mnist_benchmarks", action="store_true",
        help="Run MNIST benchmarks.")


    main(parser.parse_args())