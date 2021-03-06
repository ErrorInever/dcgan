import torch
from utils import latent_space, ones_target, zeros_target
from config.conf import cfg


def train_discriminator(discriminator, D_optimizer, criterion, real_data, real_labels, fake_data, fake_labels):
    D_optimizer.zero_grad()

    # 1. train on real data
    predict_real = discriminator(real_data).view(-1)
    loss_on_real = criterion(predict_real, real_labels)
    loss_on_real.backward()

    # 2. train on fake data
    predict_fake = discriminator(fake_data).view(-1)
    loss_on_fake = criterion(predict_fake, fake_labels)
    loss_on_fake.backward()

    D_optimizer.step()

    loss = loss_on_real + loss_on_fake

    return loss, predict_real, predict_fake


def train_generator(discriminator, G_optimizer, criterion, fake_data, real_labels):
    G_optimizer.zero_grad()
    predict = discriminator(fake_data).view(-1)
    loss = criterion(predict, real_labels)
    loss.backward()
    G_optimizer.step()
    return loss


def train_one_epoch(generator, discriminator, dataloader, G_optimizer, D_optimizer, criterion, device, epoch,
                    static_noise, metric_logger, num_sumples, freq=100):
    for n_batch, (real_batch, _) in enumerate(dataloader):
        n = real_batch.size(0)

        real_data = real_batch.to(device)
        real_labels = ones_target(n, device)
        fake_labels = zeros_target(n, device)

        # 1. train discriminator
        noise = latent_space(n, device=device)
        fake_data = generator(noise)  # [n, 3, 64, 64]
        D_loss, pred_real, pred_fake = train_discriminator(discriminator, D_optimizer, criterion,
                                                           real_data, real_labels, fake_data, fake_labels)
        # 2. train generator
        noise = latent_space(n, device=device)
        fake_data = generator(noise)
        G_loss = train_generator(discriminator, G_optimizer, criterion, fake_data, real_labels)

        # metrics
        metric_logger.log(D_loss, G_loss, pred_real, pred_fake, epoch, n_batch, len(dataloader))

        if n_batch % freq == 0:
            with torch.no_grad():
                static_fake_data = generator(static_noise)
                metric_logger.log_image(static_fake_data, num_sumples, epoch, n_batch, len(dataloader))
                metric_logger.display_status(epoch, cfg.NUM_EPOCHS, n_batch, len(dataloader), D_loss, G_loss,
                                             pred_real, pred_fake)