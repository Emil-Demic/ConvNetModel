import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--colormap', action='store_true', default=False,
                    help='Use RGB sketches instead of BW')
parser.add_argument('--save', action='store_true', default=False,
                    help='Save trained model state dict')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=5,
                    help='Number of samples in each batch.')
parser.add_argument('--lr_scheduler_step', type=int, default=5,
                    help='Number of steps for learning rate scheduler.')
parser.add_argument("--model", type=str, default='convnext',
                    help="Name of the model to use for feature extraction.")
parser.add_argument('--users', type=int, default=10,
                    help='Number of users from dataset to use')
parser.add_argument('--seed', type=int, default=42,
                    help='Seed for reproducibility.')

# model
parser.add_argument('--cls_number', type=int, default=100)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--d_ff', type=int, default=1024)
parser.add_argument('--head', type=int, default=8)
parser.add_argument('--number', type=int, default=1)
parser.add_argument('--pretrained', default=True, action='store_false')
parser.add_argument('--anchor_number', '-a', type=int, default=49)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
