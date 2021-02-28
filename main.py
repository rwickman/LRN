import argparse
from layer_manager import LayerManager

def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
        help="Load prior progress.")
    parser.add_argument("--save_dir", default="models",
        help="Directory to save models.")
    parser.add_argument("--img_size", type=int, default=256,
        help="Image square dimension size.")
    parser.add_argument("--in_channels", type=int, default=3,
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
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size.")
    parser.add_argument("--drop_rate", type=float, default=0.2,
        help="Dropout rate for state encoder .")
    
    
    main(parser.parse_args())