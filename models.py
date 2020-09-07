from torch import nn


def weights_init(m):
    """Initialized from a Normal distribution with mean=0, std=0.02
    :param m: ``nn.Module``, init model
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):

    def __init__(self, z_size, ngpu, out_channels=1):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.out_channels = out_channels
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=(4, 4,), stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self, ngpu, in_channels=1):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
