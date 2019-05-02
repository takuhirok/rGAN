import argparse
import os

import torch

from models import resnet
from utils import util
from utils.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser()
    # Seed option
    parser.add_argument('--seed', default=0, type=int)
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Genrator option
    parser.add_argument('--g_path', type=str, required=True)
    # Output options
    parser.add_argument('--out', type=str, default='samples')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    args = parser.parse_args()

    # Set up seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    # Set up generator
    g_root = os.path.dirname(args.g_path)
    g_params = util.load_params(os.path.join(g_root, 'netG_params.pkl'))
    g_iteration = int(
        os.path.splitext(os.path.basename(args.g_path))[0].split('_')[-1])
    netG = resnet.Generator(**g_params)
    netG.to(device)
    netG.load_state_dict(
        torch.load(args.g_path, map_location=lambda storage, loc: storage))
    netG.eval()

    # Set up output
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Set up visualizer
    visualizer = Visualizer(netG, device, args.out, args.num_samples,
                            netG.num_classes, args.eval_batch_size)

    # Visualize
    visualizer.visualize(g_iteration)


if __name__ == '__main__':
    main()
