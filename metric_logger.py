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

logger = logging.getLogger(__name__)


class MetricLogger:
    """Helper metric logger class"""
    def __init__(self, model_name, dataset_name, losswise_api_key, tensorboard=False):
        """
        :param model_name: ``str``, model name
        :param dataset_name: ``str``, dataset name
        :param losswise_api_key: ``str``, losswise API key
        :param tensorboard: if True uses tensorboard
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ls_api_key = losswise_api_key
        self.tensorboard = tensorboard

        self.data_subdir = "{}/{}".format(model_name, dataset_name)
        self.comment = "{}/{}".format(model_name, dataset_name)

        if self.ls_api_key:
            logger.info('Init losswise session')
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
        :param dis_loss: ``torch.autograd.Variable``, discriminator loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, discriminator predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, discriminator predicted on fake data
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, number of batches
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        step = MetricLogger._step(epoch, n_batch, num_batches)

        if self.ls_api_key:
            self.graph_loss.append(step, {'Discriminator': dis_loss, 'Generator': gen_loss})
            self.graph_acc.append(step, {'D(x)': acc_real, 'D(G(z))': acc_fake})

        if self.tensorboard:
            self.tf_logger.add_scalar('loss/dis', dis_loss, step)
            self.tf_logger.add_scalar('loss/gen', gen_loss, step)
            self.tf_logger.add_scalar('acc/D(x)', acc_real, step)
            self.tf_logger.add_scalar('acc/D(G(z))', acc_fake, step)

    def log_image(self, images, num_samples, epoch, n_batch, num_batches, normalize=True):
        """
        Logging images
        :param images: ``Torch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, number of batches
        :param normalize: if True normalize images
        """
        horizontal_grid = torchvision.utils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        step = MetricLogger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        self.tf_logger.add_image(img_name, horizontal_grid, step)

        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True, figsize=(16, 16)):
        """
        Plot and save grid images
        :param horizontal_grid: ``ndarray``, horizontal grid image
        :param grid: ``ndarray``, grid image
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param plot_horizontal: if True plot horizontal grid image
        :param figsize: ``tuple``, figure size
        """
        out_dir = os.path.join(cfg.OUT_DIR, 'data/images{}'.format(self.data_subdir))
        MetricLogger._make_dir(out_dir)

        fig = plt.figure(figsize=figsize)
        plt.imshow(np.moveaxis(horizontal_grid.detach().cpu().numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, out_dir)
        plt.close()

        fig = plt.figure()
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))
        plt.axis('off')
        self._save_images(fig, epoch, n_batch, out_dir)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, out_dir, comment=''):
        MetricLogger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))

    def close(self):
        self.session.done()
        self.tf_logger.close()

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, dis_loss, gen_loss, acc_real, acc_fake):
        """
        Display training progress
        :param epoch: ``int``, current epoch
        :param num_epochs: ``int``, numbers epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, discriminator loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, discriminator predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, discriminator predicted on fake data
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, num_epochs, n_batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))
        print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(acc_real, acc_fake))

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
