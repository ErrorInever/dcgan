import argparse
import logging
import random
import numpy as np
import torch
import time
from data import get_mnist_dataset
from torch.utils.data.dataloader import DataLoader
from config.conf import cfg
from models import *
from metric_logger import MetricLogger
from utils import latent_space
from train import train_one_epoch


def set_random_seed(val):
    """freeze random, set cuda to deterministic mode"""
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='dcGAN')
    parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
    parser.add_argument('--save_models', dest='save_models', help='save models after training', action='store_true')
    parser.add_argument('--save_state', dest='save_state', help='save state each epoch', default=10, type=int)
    parser.add_argument('--tensorboard', dest='tensorboard', help='use tensorboard', action='store_true')
    parser.add_argument('--ngpu', dest='ngpu', help='Number of GPUs availablem. Use 0 for CPU', default=0, type=int)
    parser.add_argument('--out_dir', dest='out_dir', help='Out directory', default='.', type=str)
    parser.add_argument('--resume_train', dest='resume_train', help='proceed train', action='store_true')
    parser.add_argument('--path_models', dest='path_models', help='path to models.pth', default=None, type=str)

    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    logger.setLevel(logging.WARNING)
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = parse_args()
    if args.resume_train:
        if not args.path_models:
            raise ValueError('Path to models not specified')
    if args.out_dir:
        cfg.OUT_DIR = args.out_dir
    set_random_seed(777)

    dataset = get_mnist_dataset()
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    logger.debug('Run with device %s', device)

    logger.info('Init models')

    if args.resume_train:
        pass
    else:
        print('Initialize models')
        generator = Generator(100, args.ngpu).to(device)
        # multi-gpu
        if (device.type == 'cuda') and (args.ngpu > 1):
            generator = torch.nn.DataParallel(generator, list(range(args.ngpu)))
        # init weights
        generator.apply(weights_init)

        discriminator = Discriminator(args.ngpu).to(device)
        if (device.type == 'cuda') and (args.ngpu > 1):
            discriminator = torch.nn.DataParallel(discriminator, list(range(args.ngpu)))
        discriminator.apply(weights_init)

        G_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, 0.999))
        D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, 0.999))

    criterion = nn.BCELoss()

    static_noise = latent_space(16, device=device)

    metric_logger = MetricLogger('DCGAN', 'MNIST', losswise_api_key=args.api_key, tensorboard=args.tensorboard)

    start_time = time.time()
    for epoch in range(cfg.NUM_EPOCHS):
        train_one_epoch(generator, discriminator, dataloader, G_optimizer, D_optimizer, criterion, device, epoch,
                        static_noise, metric_logger, num_sumples=16, freq=100)
        if epoch % args.save_state == 0:
            MetricLogger.checkpoint(epoch, generator, G_optimizer)
            MetricLogger.checkpoint(epoch, discriminator, D_optimizer)
    if args.save_models:
        metric_logger.save_models(generator, discriminator, cfg.NUM_EPOCHS)

    metric_logger.dump_metrics()
    total_time = time.time() - start_time
    print('Training time {}'.format(total_time))
