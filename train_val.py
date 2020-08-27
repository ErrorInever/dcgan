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
    parser.add_argument('--save_models', dest='save_models', help='save model', action='store_true')
    parser.add_argument('--tensorboard', dest='tensorboard', help='use tensorboard', action='store_true')
    parser.add_argument('--ngpu', dest='ngpu', help='Number of GPUs availablem. Use 0 for CPU', default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = parse_args()
    set_random_seed(999)

    dataset = get_mnist_dataset()
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.WORKERS)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    logger.debug('Run with device %s', device)

    logger.info('Init models')

    generator = Generator(100, 3, args.ngpu).to(device)
    # multi-gpu
    if (device.type == 'cuda') and (args.ngpu > 1):
        generator = torch.nn.DataParallel(generator, list(range(args.ngpu)))
    # init weights
    generator.apply(weights_init)

    discriminator = Discriminator(3, args.ngpu).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        discriminator = torch.nn.DataParallel(discriminator, list(range(args.ngpu)))
    discriminator.apply(weights_init)

    logger.info(generator)
    logger.info(discriminator)

    criterion = nn.BCELoss()

    static_noise = torch.randn(64, 100, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, 0.999))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA_1, 0.999))

    metric_logger = MetricLogger('DCGAN', 'MNIST', losswise_api_key=args.api_key, tensorboard=args.tensorboard)

    start_time = time.time()
    for epoch in range(cfg.NUM_EPOCHS):
        pass