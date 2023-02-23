import itertools
import numpy as np
import gpytorch.kernels
import unittest
import torch

from acgp.models.pytorch import GPUStoppedCholesky
from hyperparameter_tuning.utils.optimization_strategies.iterative_refinement import IterativeRefinement
from utils.data.load_dataset import get_train_test_dataset
from hyperparameter_tuning.utils.gpytorch.models.exact_gpr import ExactGPR
from hyperparameter_tuning.utils.gpytorch.kernel_factory import KernelFactory, KERNEL_NAME, LENGTH_SCALE
from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import ESTIMATOR, \
    ALL_POINTS, MAX_N, StoppedCholesky
from hyperparameter_tuning.utils.optimization_strategies.default_strategy import DefaultOptimizationStrategy
from run_hyper_parameter_tuning import _make_mean_func, \
    _make_noise_func
from acgp.models.custom_reverse.pytorch.log_marginal import get_custom_log_det_plus_quad
from utils.result_management.constants import BLOCK_SIZE


class MyTestCase(unittest.TestCase):
    def test_stopped_cholesky_grad_against_itself(self):
        block_size = 512
        args = {BLOCK_SIZE: block_size, ESTIMATOR: ALL_POINTS, MAX_N: 2000}

        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        N = X.shape[0]
        X = torch.tensor(X, requires_grad=False)
        y = torch.tensor(y, requires_grad=False)

        k = gpytorch.kernels.RBFKernel()
        k.train(True)

        device = "cpu"

        raw_sn2 = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float64))
        sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=torch.tensor(1e-6, dtype=torch.float64))
        #sn2 = lambda: torch.tensor(raw_sn2.data, dtype=torch.float64, requires_grad=False)
        raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        mu = _make_mean_func(raw_mu)
        #mu = lambda X: torch.zeros([X.shape[0], 1], dtype=torch.float64, requires_grad=False)

        algorithm_ = QuietChol(X, y, k, sn2, mu, args, device)

        named_variables = {n: v for n, v in
                           itertools.chain(k.named_parameters(), algorithm_.get_named_tunable_parameters())}
        #named_variables = {}
        named_variables['raw_sn2'] = raw_sn2
        named_variables['raw_mu'] = raw_mu

        closure = algorithm_.create_loss_closure()
        fx = closure()
        print(fx)
        fx.backward()
        grads = {}
        for n, v in named_variables.items():
            grads[n] = v.grad.clone()
            v.grad = torch.zeros_like(v.grad)
        algorithm = algorithm_.stopped_cholesky
        m = algorithm.subset_size
        #m = N

        print(f"subset size: {m}")
        K = k(X[:m, :]).evaluate() + sn2() * torch.eye(m)
        #K = torch.tril(K)
        #err = y[:m, :] - mu(X[:m, :])
        err = y - mu(X)
        err = err[:m, :]
        with torch.no_grad():
            L = torch.linalg.cholesky(K)
            # make sure Choleskies agree
            self.assertAlmostEqual(torch.max(torch.abs(L - torch.tril(torch.as_tensor(algorithm.A[:m, :m])))).item(), 0.)
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
            alpha, _ = torch.triangular_solve(err, L, upper=False)
            quad = torch.sum(torch.square(alpha))

            alpha, _ = torch.triangular_solve(alpha, L.T)
            self.assertAlmostEqual(torch.max(torch.abs(alpha - torch.as_tensor(algorithm.alpha0[:m, :]))).item(), 0., places=3)

        factor = 1 + (N - m) / m
        #loss = factor / 2 * (log_det + quad) + algorithm.const
        loss = factor / 2 * get_custom_log_det_plus_quad(L, alpha, result=log_det + quad, subset_size=[m]).apply(K, err) + algorithm.const
        print(loss)
        loss.backward()
        for n, v in named_variables.items():
            print(f"{n}: {grads[n]}, {v.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[n] - v.grad)).item(), 0., places=2)

    def test_stopped_cholesky_grad(self):
        block_size = 512
        args = {BLOCK_SIZE: block_size, ESTIMATOR: ALL_POINTS, MAX_N: 2000}

        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        N = X.shape[0]
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        k = gpytorch.kernels.RBFKernel()
        k.train(True)

        device = "cpu"

        raw_sn2 = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float64))
        sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=torch.tensor(1e-6, dtype=torch.float64))
        raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        mu = _make_mean_func(raw_mu)
        #mu = lambda X: torch.zeros([X.shape[0], 1], dtype=torch.float64, requires_grad=False)

        algorithm_ = QuietChol(X, y, k, sn2, mu, args, device)

        named_variables = {n: v for n, v in
                           itertools.chain(k.named_parameters(), algorithm_.get_named_tunable_parameters())}
        #named_variables = {}
        named_variables['raw_sn2'] = raw_sn2
        named_variables['raw_mu'] = raw_mu

        closure = algorithm_.create_loss_closure()
        fx = closure()
        #fx = torch.sum(algorithm.alpha0)
        print(fx)
        fx.backward()
        grads = {}
        for n, v in named_variables.items():
            grads[n] = v.grad.clone()
            v.grad = torch.zeros_like(v.grad)

        algorithm = algorithm_.stopped_cholesky
        m = algorithm.subset_size
        print(f"subset size: {m}")
        K = k(X[:m, :]).evaluate() + sn2() * torch.eye(m)
        #K = torch.tril(K)
        L = torch.linalg.cholesky(K)
        # make sure Choleskies agree
        self.assertAlmostEqual(torch.max(torch.abs(L - torch.tril(torch.as_tensor(algorithm.A[:m, :m])))).item(), 0.)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        alpha, _ = torch.triangular_solve(y[:m, :] - mu(X[:m, :]), L, upper=False)
        quad = torch.sum(torch.square(alpha))

        alpha, _ = torch.triangular_solve(alpha, L.T)
        #self.assertAlmostEqual(torch.max(torch.abs(alpha[:m-block_size, :] - torch.as_tensor(algorithm.alpha0[:m-block_size, :]))).item(), 0.)
        self.assertAlmostEqual(torch.max(torch.abs(alpha - torch.as_tensor(algorithm.alpha0[:m, :]))).item(), 0., places=5)

        factor = 1 + (N - m) / m
        loss = factor / 2 * (log_det + quad) + algorithm.const
        print(loss)
        loss.backward()
        for n, v in named_variables.items():
            print(f"{n}: {grads[n]}, {v.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[n] - v.grad)).item(), 0., places=2)

    def test_stopped_cholesky_grad_gpu(self):
        block_size = 512

        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        N = X.shape[0]
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        k = gpytorch.kernels.RBFKernel()
        k.train(True)

        raw_sn2 = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float64))
        sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=torch.tensor(1e-6, dtype=torch.float64))
        raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        mu = _make_mean_func(raw_mu)
        #mu = lambda X: torch.zeros([X.shape[0], 1], dtype=torch.float64, requires_grad=False)

        algorithm_ = GPUStoppedCholesky(X, y, k, sn2, mu, estimator=ALL_POINTS, max_n=2000, error_tolerance=IterativeRefinement().get_algorithm_tolerance, block_size=block_size)

        named_variables = {n: v for n, v in
                           itertools.chain(k.named_parameters())}
        #named_variables = {}
        named_variables['raw_sn2'] = raw_sn2
        named_variables['raw_mu'] = raw_mu

        closure = algorithm_.create_loss_closure()
        fx = closure()
        #fx = torch.sum(algorithm.alpha0)
        print(fx)
        fx.backward()
        grads = {}
        for n, v in named_variables.items():
            grads[n] = v.grad.clone()
            v.grad = torch.zeros_like(v.grad)

        algorithm = algorithm_
        m = algorithm.subset_size
        print(f"subset size: {m}")
        K = k(X[:m, :]).evaluate() + sn2() * torch.eye(m)
        #K = torch.tril(K)
        L = torch.linalg.cholesky(K)
        # make sure Choleskies agree
        self.assertAlmostEqual(torch.max(torch.abs(L - torch.tril(torch.as_tensor(algorithm.A[:m, :m])))).item(), 0.)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        alpha, _ = torch.triangular_solve(y[:m, :] - mu(X[:m, :]), L, upper=False)
        quad = torch.sum(torch.square(alpha))

        alpha, _ = torch.triangular_solve(alpha, L.T)
        #self.assertAlmostEqual(torch.max(torch.abs(alpha[:m-block_size, :] - torch.as_tensor(algorithm.alpha0[:m-block_size, :]))).item(), 0.)
        self.assertAlmostEqual(torch.max(torch.abs(alpha - torch.as_tensor(algorithm.alpha0[:m, :]))).item(), 0., places=5)

        factor = 1 + (N - m) / m
        loss = factor / 2 * (log_det + quad) + algorithm.const
        print(loss)
        loss.backward()
        for n, v in named_variables.items():
            print(f"{n}: {grads[n]}, {v.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[n] - v.grad)).item(), 0., places=2)

    def test_stopped_cholesky_grad_vs_exact(self):
        # TODO: adapt test that checks that with increasing precision gradients match better
        kf = KernelFactory()
        #kf.add_tags_to_experiment(args)
        kernel_name = kf.get_available_kernel_functions()[0]
        args = {KERNEL_NAME: kernel_name, LENGTH_SCALE: 0., BLOCK_SIZE: 256,
                ESTIMATOR: ALL_POINTS, MAX_N: 2000}

        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=0)
        # X = X[:100, :]
        # y = y[:X.shape[0], :]
        # log(level=ERROR, msg="using only subset!")
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)

        k = kf.create(args, X)
        k.train(True)

        device = "cpu"

        raw_sn2 = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float64))
        sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=torch.tensor(1e-6, dtype=torch.float64))
        raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        mu = _make_mean_func(raw_mu)

        algorithm_ = QuietChol(X, y, k, sn2, mu, args, device,
                              optimization_strategy=DefaultOptimizationStrategy(algorithm_tolerance=0.))
        gt_algo = ExactGPR(X, y, k, sn2, mu, args, device)

        named_variables = {n: v for n, v in
                           itertools.chain(k.named_parameters(), algorithm_.get_named_tunable_parameters())}
        named_variables['raw_sn2'] = raw_sn2
        named_variables['raw_mu'] = raw_mu

        closure = algorithm_.create_loss_closure()
        fx = closure()
        #fx = torch.sum(algorithm.alpha0)
        print(fx)
        fx.backward()
        grads = {}
        for n, v in named_variables.items():
            grads[n] = v.grad.clone()
            v.grad = torch.zeros_like(v.grad)

        closure = gt_algo.create_loss_closure()
        fx = closure()
        #fx = torch.sum(algorithm.alpha0)
        print(fx)
        fx.backward()

        for n, v in named_variables.items():
            print(f"{n}: {grads[n]}, {v.grad}")
            self.assertAlmostEqual(torch.max(torch.abs(grads[n] - v.grad)).item(), 0., places=2)

    def test_stopped_cholesky_prediction_vs_exact(self):
        kf = KernelFactory()
        #kf.add_tags_to_experiment(args)
        kernel_name = kf.get_available_kernel_functions()[0]
        args = {KERNEL_NAME: kernel_name, LENGTH_SCALE: 0., BLOCK_SIZE: 2000,
                ESTIMATOR: ALL_POINTS, MAX_N: 2000}

        X, y, X_test, y_test = get_train_test_dataset("toydata2000", seed=np.random.randint(32000))
        # X = X[:100, :]
        # y = y[:X.shape[0], :]
        # log(level=ERROR, msg="using only subset!")
        X = torch.as_tensor(X)
        y = torch.as_tensor(y)
        X_test = torch.as_tensor(X_test)

        k = kf.create(args, X)
        k.train(True)

        device = "cpu"

        raw_sn2 = torch.nn.Parameter(torch.tensor(1e-3, dtype=torch.float64))
        sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=torch.tensor(1e-6, dtype=torch.float64))
        raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        mu = _make_mean_func(raw_mu)

        algorithm = QuietChol(X, y, k, sn2, mu, args, device)
        gt_algo = ExactGPR(X, y, k, sn2, mu, args, device)

        closure = algorithm.create_loss_closure()
        fx = closure()

        closure = gt_algo.create_loss_closure()
        fx_gt = closure()

        # The difference in numerical precision can be large in absolute precision.
        #self.assertAlmostEqual(torch.max(torch.abs(fx - fx_gt)).item(), 0., places=5)
        self.assertAlmostEqual(fx.item(), fx_gt.item(), places=3)
        #np.testing.assert_array_almost_equal(fx.item(), fx_gt.item())

        m, v = algorithm.get_posterior(X_test)
        m_gt, v_gt = gt_algo.get_posterior(X_test)
        np.testing.assert_array_almost_equal(m.numpy(), m_gt.numpy())
        np.testing.assert_array_almost_equal(v.numpy(), v_gt.numpy())


class QuietChol(StoppedCholesky):
    def set_tag(self, *args):
        pass


if __name__ == '__main__':
    unittest.main()
