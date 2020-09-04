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


class Discriminator(nn.Module):

    def __init__(self, in_channels, ngpu):
        """
        :param in_channels: ``int``, in channels (out_channels from Generator)
        :param ngpu: number of GPUs available
        """
        super().__init__()
        self.ngpu = ngpu
        self.in_channels = in_channels

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.body = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_size, out_channels, ngpu):
        """
        :param z_size: ``int``, input vector size (latent space)
        :param out_channels: ``int``, out channels
        :param ngpu: number of GPUs available
        """
        super().__init__()
        self.ngpu = ngpu
        self.z_size = z_size
        self.out_channels = out_channels

        self.head = nn.Sequential(
            nn.ConvTranspose2d(z_size, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return x
