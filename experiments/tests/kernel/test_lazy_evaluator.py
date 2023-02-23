import torch

import gpytorch.kernels
import unittest
import numpy as np
from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import StoppedCholesky
from acgp.models.pytorch import GPUStoppedCholesky, CPUStoppedCholesky

from utils.mem_efficient_kernels.isotropic_kernel import RBF
from utils.kernel_evaluator import get_kernel_evaluator


class MyTestCase(unittest.TestCase):
    def test_kernel_evaluator(self):
        n = 1299
        D = 3
        sn2 = 1e-4
        block_size = 5 * 64
        X = np.asfortranarray(np.random.randn(n, D))
        k = RBF().initialize()
        ke = get_kernel_evaluator(X, lambda *args: k.K(*args), sn2)
        A = np.zeros([n, n], order='F')
        ke(A, 0, block_size, 0, block_size)
        for i in range(block_size, n, block_size):
            advance = min(block_size, n-i)
            ke(A, i, advance, i, advance)
            ke(A, i, advance, 0, i)

        np.testing.assert_array_almost_equal(np.tril(A), np.tril(k.K(X) + sn2 * np.eye(n)))

    def test_hyperparametertuning_kernel_evaluator(self):
        n = 3299
        D = 3
        sn2 = torch.tensor(1e-4, dtype=torch.float64)
        block_size = 5 * 64
        X = torch.as_tensor(np.asfortranarray(np.random.randn(n, D)))
        K = torch.zeros([n, n], dtype=torch.float64)
        k = gpytorch.kernels.RBFKernel()
        k.train()
        ke = CPUStoppedCholesky._get_kernel_evaluator(X, lambda *args: k(*args).evaluate(), sn2, K)
        A = np.zeros([n, n], order='F')
        ke(A, 0, block_size, 0, block_size)
        for i in range(block_size, n, block_size):
            advance = min(block_size, n-i)
            ke(A, i, advance, i, advance)
            ke(A, i, advance, 0, i)

        max_diff = torch.max(torch.abs(torch.tril(K - (k(X).evaluate() + sn2 * torch.eye(n, dtype=torch.float64))))).item()
        print(max_diff)
        self.assertAlmostEqual(max_diff, 0.)
        self.assertAlmostEqual(torch.max(torch.abs(torch.tril(torch.as_tensor(A) - k(X).evaluate() - sn2 * torch.eye(n)))).item(), 0.)


if __name__ == '__main__':
    unittest.main()
