import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--save', action='store_true', default=False,
                    help='Save trained model state dict')

parser.add_argument('--val_unseen', action='store_true', default=False,
                    help='Use unseen user train/val split')

parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')

parser.add_argument('--temp', type=float, default=0.05,
                    help='Temperature parameter for InfoNCE loss.')

parser.add_argument('--batch_size', type=int, default=5,
                    help='Number of samples in each batch.')

parser.add_argument("--backbone", type=str, default='convnext',
                    help="Name of the backbone to use for feature extraction.")

parser.add_argument("--dataset", type=str, default='fscoco',
                    help="Name of the dataset to use.")

parser.add_argument('--seed', type=int, default=42,
                    help='Seed for reproducibility.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
