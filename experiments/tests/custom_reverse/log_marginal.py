import gpytorch.kernels
import unittest
import torch
from utils.data.load_dataset import get_train_test_dataset
from acgp.models.custom_reverse.pytorch.log_marginal import get_custom_log_det_plus_quad


class MyTestCase(unittest.TestCase):
    def test_something(self):
        N = 300
        D = 10
        sn2 = 1e-2
        torch.random.manual_seed(0)
        X = torch.randn([N, D], dtype=torch.float64)
        y = torch.randn([N, 1], dtype=torch.float64)
        # X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        # N, D = X.shape
        # sn2 = 1e-3

        ls = torch.nn.Parameter(torch.ones(D, dtype=torch.float64))
        amp = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        mu = torch.nn.Parameter(0. * torch.ones(1, dtype=torch.float64))
        grads = {ls: None, amp: None, mu: None}

        def _get_neg_dist(X: torch.Tensor):
            Xsq = torch.sum(torch.square(X), dim=1)
            K = 2. * X @ X.T
            K = K - Xsq[:, None]
            K = K - Xsq[None, :]
            K = K
            return K

        def k(X):
            return torch.exp(_get_neg_dist(X / ls)) * torch.square(amp)

        err = y - mu
        K = k(X) + sn2 * torch.eye(N, dtype=torch.float64)

        L = torch.linalg.cholesky(K)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        #alpha = torch.cholesky_solve(err, L)
        #result = log_det + err.T @ alpha
        alpha0 = torch.triangular_solve(err, L, upper=False).solution
        quad = torch.sum(torch.square(alpha0))
        result = log_det + quad
        alpha = torch.triangular_solve(alpha0, L.T).solution

        def post_result_f(r):
            return r / 2 + N / 2 * torch.log(torch.tensor(2 * 3.14, dtype=torch.float64))

        #result.backward()
        result_ = post_result_f(result)
        result_.backward()
        for g in grads.keys():
            grads[g] = g.grad.clone()
            g.grad = torch.zeros_like(g.grad)

        err = y - mu
        K = k(X) + sn2 * torch.eye(N, dtype=torch.float64)

        fx = get_custom_log_det_plus_quad(L, alpha, result, subset_size=[N]).apply(K, err)
        fx = post_result_f(fx)
        fx.backward()
        for g in grads.keys():
            print(f"{grads[g]}, {g.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[g] - g.grad)).item(), 0., places=2)

    def test_with_gpt_kernel(self):
        # N = 300
        # D = 10
        # sn2 = 1e-2
        # #torch.random.manual_seed(0)
        # X = torch.randn([N, D], dtype=torch.float64)
        # y = torch.randn([N, 1], dtype=torch.float64)
        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        N, D = X.shape
        sn2 = 1e-3

        mu = torch.zeros(1, dtype=torch.float64, requires_grad=False)
        k_ = gpytorch.kernels.RBFKernel()
        k = lambda *args: k_(*args).evaluate()
        grads = {p: None for p in k_.parameters()}

        err = y - mu
        K = k(X) + sn2 * torch.eye(N, dtype=torch.float64)

        L = torch.linalg.cholesky(K)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        #alpha = torch.cholesky_solve(err, L)
        #result = log_det + err.T @ alpha
        alpha0 = torch.triangular_solve(err, L, upper=False).solution
        quad = torch.sum(torch.square(alpha0))
        result = log_det + quad
        alpha = torch.triangular_solve(alpha0, L.T).solution

        def post_result_f(r):
            return r / 2 + N / 2 * torch.log(torch.tensor(2 * 3.14, dtype=torch.float64))

        #result.backward()
        result_ = post_result_f(result)
        print(result_)
        result_.backward()
        for g in grads.keys():
            grads[g] = g.grad.clone()
            g.grad = torch.zeros_like(g.grad)

        err = y - mu
        K = k(X) + sn2 * torch.eye(N, dtype=torch.float64)

        fx = get_custom_log_det_plus_quad(L, alpha, result, subset_size=[N]).apply(K, err)
        fx = post_result_f(fx)
        print(fx)
        fx.backward()
        for g in grads.keys():
            print(f"{grads[g]}, {g.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[g] - g.grad)).item(), 0., places=2)


if __name__ == '__main__':
    unittest.main()
