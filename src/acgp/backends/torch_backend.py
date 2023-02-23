import torch

from acgp.backends.numpy_backend import NumpyBackend


class TorchBackend(NumpyBackend):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.pi = torch.tensor(self.pi, device=self.device)

    def sum(self, *args):
        return torch.sum(*args)

    def log(self, *args):
        return torch.log(*args)

    def diag(self, *args, k=0):
        return torch.diag(*args, diagonal=k).reshape(-1, 1)

    def square(self, *args):
        return torch.square(*args)

    def min(self, *args):
        return torch.min(*args)

    def max(self, *args):
        return torch.max(*args)

    def abs(self, *args):
        return torch.abs(*args)

    def ones(self, *args):
        return torch.ones(*args, device=self.device)

    def sign(self, *args):
        return torch.sign(*args)

    def floor(self, *args):
        return torch.floor(*args)

    def sqrt(self, *args):
        return torch.sqrt(*args)

    def eye(self, *args):
        return torch.eye(*args, device=self.device)
