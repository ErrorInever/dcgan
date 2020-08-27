from torch import nn


class Generator(nn.Module):

    def __init__(self, z_size, out_channels, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.z_size = z_size
        self.out_channels = out_channels

        self.head = nn.Sequential(
            nn.ConvTranspose2d(z_size, 1024, 4, 0, bias=False),
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