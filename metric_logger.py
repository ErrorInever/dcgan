import losswise
import torch
import os
import errno
import numpy as np
import torchvision
import logging
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from IPython import display
from config.conf import cfg


class Logger:
    """Helper logger class"""
    def __init__(self, model_name, dataset_name, losswise_api_key, tensorboard=False, log_level=logging.DEBUG):
        """
        """
        self.p_logger = logging.getLogger('main_logger')
        self.ch = logging.StreamHandler()
        self.set_p_logger_level(log_level)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.ch.setFormatter(formatter)
        self.p_logger.addHandler(self.ch)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ls_api_key = losswise_api_key
        self.tensorboard = tensorboard

        self.data_subdir = "{}/{}".format(model_name, dataset_name)
        self.comment = "{}/{}".format(model_name, dataset_name)

        if self.ls_api_key:
            self.p_logger.info('Init losswise session')
            losswise.set_api_key(self.ls_api_key)
            self.session = losswise.Session(
                    tag=model_name,
                    max_iter=cfg.NUM_EPOCH,
                    track_git=False
                    )
            self.graph_loss = self.session.graph('loss', kind='min')
            self.graph_acc = self.session.graph('accuracy', kind='max')

        if self.tensorboard:
            self.tf_logger = SummaryWriter(comment=self.comment)

    def log(self, dis_loss, gen_loss, acc_real, acc_fake, epoch, n_batch, num_batches):
        """
        Logging training values
        :param dis_loss: discriminator loss
        :param gen_loss: generator loss
        :param acc_real: accuracy on real data
        :param acc_fake: accuracy on fake data
        :param epoch: current epoch
        :param n_batch: current batch
        :param num_batches: number batches
        """
        pass

    def set_p_logger_level(self, log_level):
        """
        :param log_level:
        :return:
        """
        self.p_logger.setLevel(log_level)
        self.ch.setLevel(log_level)
