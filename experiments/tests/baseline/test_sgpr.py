import math
import unittest

import gpytorch
import torch
from gpytorch.kernels import MaternKernel

from utils.data.load_dataset import get_train_test_dataset
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import NativeVariationalGPR, NUM_INDUCING_INPUTS, SELECTION_SCHEME, \
    CONDITIONAL_VARIANCE, OPTIMIZE_INDUCING_INPUTS, VariationalGPR


class SGPRTestCase(unittest.TestCase):
    def test_equivalence_naive_implementation(self):
        X, y, Xtest, ytest = get_train_test_dataset("wilson_pumadyn32nm", seed=0)
        dtype = torch.float64
        torch.set_default_dtype(dtype)
        X = torch.as_tensor(X, dtype=dtype)
        y = torch.as_tensor(y, dtype=dtype)
        Xtest = torch.as_tensor(Xtest, dtype=dtype)
        ytest = torch.as_tensor(ytest, dtype=dtype)

        k = MaternKernel()
        sn2 = lambda: 1e-4 * torch.ones(1, dtype=dtype)
        mu = lambda X: torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        args = {
            NUM_INDUCING_INPUTS: 32,
            SELECTION_SCHEME: CONDITIONAL_VARIANCE,
            OPTIMIZE_INDUCING_INPUTS: False,
        }
        sgpr = NativeVariationalGPR(X, y, k, sn2, mu, args)
        Z = -(sgpr.create_loss_closure()())  # minus because the loss returns the negative ELBO
        #m, v = sgpr.get_posterior(Xtest)

        Kmm = k(sgpr.inducing_points).evaluate() + 1e-8 * torch.eye(sgpr.inducing_points.shape[0])
        L = torch.cholesky(Kmm)
        A = torch.triangular_solve(k(sgpr.inducing_points, X).evaluate(), L, upper=False)[0]
        Qnn = A.transpose(0, 1) @ A
        Qnn += sn2() * torch.eye(X.shape[0])
        #mvn = torch.distributions.MultivariateNormal(loc=mu(X), covariance_matrix=Qnn)
        #Z_gt = mvn.log_prob(y) - torch.sum(k(X, diag=True) - Qnn.diag()) / sn2()
        L = torch.cholesky(Qnn)
        quad = torch.triangular_solve(y-mu(X), L, upper=False)[0].square().sum()
        print(f"quad: {quad}")
        logdet = L.diag().log().sum()
        print(f"logdet: {logdet}")
        trace_diff = torch.sum((k(X, diag=True)+sn2()) - Qnn.diag()) / sn2()
        print(f"trace_diff: {trace_diff}")
        c = X.shape[0] / 2 * torch.log(2 * torch.tensor(math.pi))
        print(f"c: {c}")
        Z_gt = -quad / 2 - logdet - trace_diff / 2 - sgpr.const
        self.assertAlmostEqual(Z_gt.item(), Z.item())


if __name__ == '__main__':
    unittest.main()
