import numpy as np


class NumpyBackend:
    """
    Wrapper for standard functions.
    """
    def __init__(self):
        self.pi = np.pi

    def sum(self, *args):
        return np.sum(*args)

    def log(self, *args):
        return np.log(*args)

    def diag(self, *args, **kwargs):
        return np.diag(*args, **kwargs).reshape(-1, 1)

    def square(self, *args):
        return np.square(*args)

    def min(self, *args):
        return np.min(*args)

    def max(self, *args):
        return np.max(*args)

    def abs(self, *args):
        return np.abs(*args)

    def ones(self, *args):
        return np.ones(*args)

    def sign(self, *args):
        return np.sign(*args)

    def floor(self, *args):
        return np.floor(*args)

    def sqrt(self, *args):
        return np.sqrt(*args)

    def eye(self, *args):
        return np.eye(*args)
