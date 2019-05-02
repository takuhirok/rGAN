import argparse
import functools
import os
import time

import torch
import torch.optim as optim
from torch.backends import cudnn
import torchvision.transforms as transforms

import datasets
import trainers
from models import common, resnet
from utils import util
from utils.logger import Logger
from utils.visualizer import Visualizer


def main():
    # For fast training
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--data_noise_type', type=str, default='symmetric')
    parser.add_argument('--data_noise_rate', type=float, default=0.)
    parser.add_argument('--data_seed', type=int, default=12345)
    parser.add_argument('--num_workers', type=int, default=4)
    # Model options
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_channels', type=int, default=256)
    parser.add_argument('--g_spectral_norm', type=int, default=0)
    parser.add_argument('--d_channels', type=int, default=128)
    parser.add_argument('--d_dropout', type=int, default=0)
    parser.add_argument('--d_spectral_norm', type=int, default=1)
    parser.add_argument('--d_pooling', type=str, default='sum')
    # Training options
    parser.add_argument('--trainer', type=str, default='rcgan')
    parser.add_argument('--gan_loss', type=str, default='hinge')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--g_bs_multiple', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--num_critic', type=int, default=5)
    parser.add_argument('--lambda_gp', type=float, default=0.)
    parser.add_argument('--lambda_ct', type=float, default=0.)
    parser.add_argument('--lambda_cls_g', type=float, default=0.1)
    parser.add_argument('--lambda_cls_d', type=float, default=1.)
    parser.add_argument('--factor_m', type=float, default=0.)
    parser.add_argument('--model_noise_type', type=str, default='symmetric')
    parser.add_argument('--model_noise_rate', type=float, default=0.)
    parser.add_argument('--num_iterations', type=int, default=100000)
    parser.add_argument('--num_iterations_decay', type=int, default=100000)
    # Output options
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--snapshot_interval', type=int, default=5000)
    parser.add_argument('--visualize_interval', type=int, default=5000)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    args = parser.parse_args()
    args.g_spectral_norm = bool(args.g_spectral_norm)
    args.d_dropout = bool(args.d_dropout)
    args.d_spectral_norm = bool(args.d_spectral_norm)

    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    else:
        device = torch.device('cpu')

    # Set up dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
        Dataset = functools.partial(datasets.CIFAR10NoisyLabels,
                                    noise_type=args.data_noise_type,
                                    noise_rate=args.data_noise_rate,
                                    seed=args.data_seed)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        Dataset = functools.partial(datasets.CIFAR100NoisyLabels,
                                    noise_type=args.data_noise_type,
                                    noise_rate=args.data_noise_rate,
                                    seed=args.data_seed)

    def normalize(x):
        x = 2 * ((x * 255. / 256.) - .5)
        x += torch.zeros_like(x).uniform_(0, 1. / 128)
        return x

    dataset = Dataset(root=args.dataroot,
                      train=True,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Lambda(normalize)
                      ]))

    iterator = util.InfDataLoader(dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  drop_last=True)

    # Set up output
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    util.print_args(args, os.path.join(args.out, 'args.txt'))

    # Set up noise transition matrix
    if args.model_noise_rate > 0:
        T = dataset.T(args.model_noise_type, args.model_noise_rate).to(device)
        noise2clean = functools.partial(util.noise2clean, T=T)
    else:
        T = None
        noise2clean = lambda x: x

    # Set up models
    g_params = {
        'latent_dim': args.latent_dim,
        'num_classes': args.num_classes,
        'channels': args.g_channels,
        'image_size': args.image_size,
        'spectral_norm': args.g_spectral_norm
    }
    d_params = {
        'num_classes': args.num_classes,
        'channels': args.d_channels,
        'dropout': args.d_dropout,
        'spectral_norm': args.d_spectral_norm,
        'pooling': args.d_pooling
    }
    if args.trainer == 'rcgan':
        netG = resnet.Generator(**g_params)
        netD = resnet.ProjectionDiscriminator(**d_params)
    elif args.trainer == 'racgan':
        netG = resnet.Generator(**g_params)
        netD = resnet.ACGANDiscriminator(**d_params)

    util.save_params(g_params, os.path.join(args.out, 'netG_params.pkl'))
    util.save_params(d_params, os.path.join(args.out, 'netD_params.pkl'))
    netG.to(device)
    netD.to(device)
    netG.apply(common.weights_init)
    netD.apply(common.weights_init)
    util.print_network(netG, 'G', os.path.join(args.out, 'netG_arch.txt'))
    util.print_network(netD, 'D', os.path.join(args.out, 'netD_arch.txt'))

    # Set up optimziers
    optimizerG = optim.Adam(netG.parameters(),
                            lr=args.g_lr,
                            betas=(args.beta1, args.beta2))
    optimizerD = optim.Adam(netD.parameters(),
                            lr=args.d_lr,
                            betas=(args.beta1, args.beta2))

    # Set up learning rate schedulers
    def lr_lambda(iteration):
        if args.num_iterations_decay > 0:
            lr = 1.0 - max(0,
                           (iteration + 1 -
                            (args.num_iterations - args.num_iterations_decay)
                            )) / float(args.num_iterations_decay)
        else:
            lr = 1.0
        return lr

    lr_schedulerG = optim.lr_scheduler.LambdaLR(optimizerG,
                                                lr_lambda=lr_lambda)
    lr_schedulerD = optim.lr_scheduler.LambdaLR(optimizerD,
                                                lr_lambda=lr_lambda)

    # Set up trainer
    trainter_params = {
        'iterator': iterator,
        'models': (netG, netD),
        'optimizers': (optimizerG, optimizerD),
        'gan_loss': args.gan_loss,
        'lr_schedulers': (lr_schedulerG, lr_schedulerD),
        'batch_size': args.batch_size,
        'g_bs_multiple': args.g_bs_multiple,
        'num_critic': args.num_critic,
        'factor_m': args.factor_m,
        'noise2clean': noise2clean,
        'device': device
    }
    if args.trainer == 'rcgan':
        Trainer = trainers.rcGANTrainer
        trainter_params.update({'lambdas': (args.lambda_gp, args.lambda_ct)})
    elif args.trainer == 'racgan':
        Trainer = trainers.rACGANTrainer
        trainter_params.update({
            'lambdas': (args.lambda_gp, args.lambda_ct, args.lambda_cls_g,
                        args.lambda_cls_d),
            'T':
            T
        })
    trainer = Trainer(**trainter_params)

    # Set up visualizer and logger
    visualizer = Visualizer(netG, device, args.out, args.num_samples,
                            netG.num_classes, args.eval_batch_size)
    logger = Logger(args.out, 'loss')

    # Train
    while trainer.iteration < args.num_iterations:
        iter_start_time = time.time()
        trainer.update()

        if (args.display_interval > 0
                and trainer.iteration % args.display_interval == 0):
            t = (time.time() - iter_start_time) / args.batch_size
            logger.log(trainer.iteration, trainer.get_current_loss(), t)

        if (args.snapshot_interval > 0
                and trainer.iteration % args.snapshot_interval == 0):
            torch.save(
                netG.state_dict(),
                os.path.join(args.out, 'netG_iter_%d.pth' % trainer.iteration))
            torch.save(
                netD.state_dict(),
                os.path.join(args.out, 'netD_iter_%d.pth' % trainer.iteration))

        if (args.visualize_interval > 0
                and trainer.iteration % args.visualize_interval == 0):
            visualizer.visualize(trainer.iteration)


if __name__ == '__main__':
    main()
