import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision import transforms, datasets


def get_mnist_dataset():
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


def show_grid_data(batch):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(batch[0][:64], padding=2, normalize=True), (1, 2, 0)))
