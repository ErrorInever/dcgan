import torch


def latent_space(n, device):
    return torch.randn(n, 100, 1, 1, device=device)


def ones_target(size, device):
    """
    Tensor containing ones, with shape = size
    """
    return torch.full(size, 1., dtype=torch.float, device=device)


def zeros_target(size, device):
    """
    Tensor containing zeros, with shape = size
    """
    return torch.full(size, 0., dtype=torch.float, device=device)
