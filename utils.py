import torch


def latent_space(n, device):
    return torch.randn(n, 100, 1, 1, device=device)


def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = torch.ones(size, 1)
    return data


def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = torch.zeros(size, 1)
    return data
