import math
import gpytorch.kernels
import logging
import numpy as np
import torch
from typing import Callable

from acgp.models.custom_reverse.pytorch.log_marginal import get_custom_log_det_plus_quad

try:
    from acgp.blas_wrappers.openblas.openblas_wrapper import OpenBlasWrapper as CPUBlasWrapper
except Exception as e:
    logging.exception(e)
    from acgp.blas_wrappers.numpy_blas_wrapper import NumpyBlasWrapper as CPUBlasWrapper

from acgp.blas_wrappers.torch_wrapper import TorchWrapper as GPUBlasWrapper
from acgp.meta_cholesky import MetaCholesky
from acgp.hooks.stop_hook import StopHook
from acgp.backends.numpy_backend import NumpyBackend
from acgp.backends.torch_backend import TorchBackend


DEFAULT = "default"
ALL_POINTS = "all_points"
estimators = [DEFAULT, ALL_POINTS]


class CPUStoppedCholesky():
    def __init__(self, X: torch.Tensor, y: torch.Tensor, k: gpytorch.kernels.Kernel, sn2: Callable, mu: Callable,
                 estimator: str, max_n: int, error_tolerance: Callable, block_size: int, initial_block_size: int=None):
        """

        :param X:
            inputs, max_n x D tensor
        :param y:
            targets, max_n x 1 tensor
        :param k:
            kernel function
        :param sn2:
            a function(!) that returns the noise
        :param mu:
            prior mean function
        :param estimator:
            string that is an element of the estimators list
        :param max_n:
            maximal dataset size that we are willing to process
        :param error_tolerance:
            a function that returns a float between 0 and 1
        :param block_size:
            number of points that are processed in parallel
            also determines the sample size of the estimators
        :param initial_block_size:
            how many points to process before even considering stopping
        """
        self.estimator = estimator
        self.X = X
        self.device = X.device
        self.y = y
        self.k = k
        self.sn2 = sn2
        self.mu = mu
        self.block_size = block_size
        if initial_block_size is None:
            initial_block_size = block_size
        self.initial_block_size = initial_block_size
        self.max_n = max_n
        if max_n < X.shape[0]:
            self.Xsub = X[:max_n, :]
            self.ysub = y[:max_n, :]
        else:
            self.Xsub = X
            self.ysub = y
        self.error_tolerance = error_tolerance

        self.last_iter = None
        self.last_advance = None
        self.alpha0 = None
        self.subset_size = 0
        self.blaswrapper = CPUBlasWrapper()
        self.backend = NumpyBackend()
        #self.bound_backend = TorchBackend(device=self.device)
        self.const = self.X.shape[0] / 2 * torch.log(2 * torch.tensor(math.pi, requires_grad=False, device=self.device, dtype=self.y.dtype))
        self.A = None
        self.K = None  # buffer for the kernel matrix
        self._setup_buffers()  # in a separate method for easier exchange

    def _setup_buffers(self):
        N = min(self.X.shape[0], self.max_n)
        self.A = np.zeros([N, N], order='F')
        self.K = np.zeros_like(self.A)
        self.alpha0 = np.zeros([N, 1])  # no Fortran here!

    def create_loss_closure(self):
        N = self.X.shape[0]
        maxN = self.A.shape[0]
        chol = MetaCholesky(block_size=self.block_size, initial_block_size=self.initial_block_size,
                            blaswrapper=self.blaswrapper)

        L = torch.as_tensor(self.A)
        alpha = torch.as_tensor(self.alpha0)
        det_plus_quad = torch.zeros(1, dtype=self.X.dtype, device=self.X.device, requires_grad=False)
        subset_size_ = [self.subset_size]
        custom = get_custom_log_det_plus_quad(L, alpha, det_plus_quad, subset_size_)

        if self.estimator == DEFAULT:
            iter_det_plus_quad_ = torch.zeros_like(det_plus_quad)
            iter_ = [0]
            iter_custom = get_custom_log_det_plus_quad(L, alpha, iter_det_plus_quad_, iter_)

        def loss_closure():
            # the step below is necessary to convince torch that we compute the gradient of a new matrix
            K = self._get_K()
            err = self.ysub - self.mu(self.Xsub)
            # get a new copy of y-mu(X) to overwrite
            # TODO: we could even be a tiny bit more efficient here by copying y-mu(X) only if asked for
            self._make_err_copy(err)  # writes to alpha0
            hook = self._get_hook()
            k_func = lambda *args: self.k(*args).evaluate()
            chol.run_configuration(self.A, self.alpha0, kernel_evaluator=self._get_kernel_evaluator(X=self.X, k=k_func,
                                                                                                    sn2=self.sn2(), K=K),
                                   hook=hook)

            iter = hook.iteration  # where we stopped
            advance = min(self.block_size, maxN - iter)  # number of elements missing to complete iteration
            subset_size = iter + advance  # number of datapoints we touched
            subset_size_[0] = subset_size

            # write things in self so they can be logged
            self.last_iter = iter
            self.last_advance = advance
            self.subset_size = subset_size

            if advance > 0:
                # finish the last step of the Cholesky
                K_ = self.A[iter:iter + advance, iter:iter + advance]
                y_ = self.alpha0[iter:iter+advance, :]
                self.blaswrapper.in_place_chol(K_)
                self.blaswrapper.solve_triangular_inplace(K_, y_, transpose_b=False, lower=True)

            # log-determinant and quadratic form of the SUBSET
            with torch.no_grad():
                L_ = self.A[:subset_size, :subset_size]
                L = torch.as_tensor(L_)
                log_sub_det = 2 * torch.sum(torch.log(torch.diag(L)))

                alpha0_ = self.alpha0[:subset_size, :]
                alpha = torch.as_tensor(alpha0_)
                sub_quad = torch.sum(torch.square(alpha))
                # here we pass the solution to our custom backward pass wrapper
                det_plus_quad[0] = log_sub_det + sub_quad
                # we don't need alpha0 anymore--we can overwrite it for the mean prediction
                # actually, this is also necessary for the custom backward pass
                self.blaswrapper.solve_triangular_inplace(L_, alpha0_, transpose_b=False, transpose_a=True, lower=True)

            K = K[:subset_size, :subset_size]
            err = err[:subset_size, :]
            # since we only evaluated the lower triangular part we need to inform torch about the upper part
            # TODO: (even though this shouldn't be necessary...)
            # TODO: this step actually contributes one or two seconds to our run time
            K = K + torch.tril(K, diagonal=-1).T
            # here we do the gradient computation more efficiently
            sub_log_det_plus_quad = custom.apply(K, err)

            if self.estimator == ALL_POINTS:
                # I suspect that for the gradient the bound estimator favors too much the last processed datapoints.
                # Hence, let's experiment with the following estimators which incorporate ALL datapoints equally.
                # However, it also appears that this estimator can give the linesearch a bit more trouble.
                factor = 1 + (N - subset_size) / subset_size
                #log_det_plus_quad = factor * sub_log_det_plus_quad
                # we want to minimize the NEGATIVE log-likelihood, hence no minus!
                return factor / 2 * sub_log_det_plus_quad + self.const
            elif self.estimator == DEFAULT:
                if advance > 0:
                    # TODO: probably this implementation can be done (A LOT) faster
                    # TODO: However, it seems not trivial to derive the backward pass for this
                    iter_log_det = 2 * torch.sum(torch.log(torch.diag(L[:iter,:iter])))
                    iter_quad = torch.sum(torch.square(alpha[:iter, :]))
                    iter_det_plus_quad_[0] = iter_log_det + iter_quad
                    iter_[0] = iter
                    iter_det_plus_quad = iter_custom.apply(K, err)
                    estimate = iter_det_plus_quad / 2
                    estimate = estimate + (N - subset_size) / advance * (sub_log_det_plus_quad - iter_det_plus_quad) / 2
                else:
                    estimate = sub_log_det_plus_quad / 2
                return estimate + self.const

        return loss_closure

    def get_posterior(self, X_star, full_posterior=False):
        # Predictive-posterior computation from GP Book / Rasmussen et al. 2006 (pp. 19)
        with torch.no_grad():
            L = torch.as_tensor(self.A[:self.subset_size, :self.subset_size])
            # the alpha0 is NOT the alpha from the Rasmussen book
            v = self.k(self.X[:self.subset_size, :], X_star).evaluate()
            f_m_star = self.mu(X_star) + v.T @ torch.as_tensor(self.alpha0[:self.subset_size, :])
            torch.triangular_solve(v, L, upper=False, out=(v, L))
            if full_posterior:
                f_v_star = self.k(X_star).evaluate() - v.T @ v
            else:
                # it appears that when diag=True then the returned tensor is not lazy...
                f_v_star = self.k(X_star, diag=True) - torch.sum(torch.square(v), dim=[0])
                f_v_star = torch.reshape(f_v_star, [-1, 1])
            return f_m_star, f_v_star

    @classmethod
    def _get_kernel_evaluator(cls, X, k, sn2, K):
        def kernel_evaluator(A, i0, i1, j0, j1):
            if i0 == j0 and i1 == j1:
                # TODO: is there a better way to fill the diagonal? Allocating a whole identity matrix seems expensive
                # it appears that for the gradient computation we have to apply tril already here
                K[i0:i0 + i1, i0:i0 + i1] = torch.tril(k(X[i0:i0 + i1, :]) + sn2 * torch.eye(i1, device=K.device, dtype=X.dtype))
                # copy values into designated array
                A[i0:i0 + i1, j0:j0 + j1] = K[i0:i0 + i1, j0:j0 + j1].detach().numpy()
            elif j1 <= i0:
                K[i0:i0 + i1, j0:j0 + j1] = k(X[i0:i0 + i1, :], X[j0:j0 + j1, :])
                # copy values into designated array
                # Fortran order seems to be preserved (or the tests wouldn't succeed)
                A[i0:i0 + i1, j0:j0 + j1] = K[i0:i0 + i1, j0:j0 + j1].detach().numpy()
            else:
                raise RuntimeError("This case should not occur")

        return kernel_evaluator

    def _make_err_copy(self, err):
        self.alpha0[:] = err.detach().numpy()[:]

    def _get_K(self):
        # this will avoid a copy
        return torch.as_tensor(self.K)

    def _get_hook(self):
        return StopHook(N=self.X.shape[0], min_noise=self.sn2().item(),
                        relative_tolerance=self.error_tolerance(), backend=self.backend)


class GPUStoppedCholesky(CPUStoppedCholesky):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blaswrapper = GPUBlasWrapper()
        self.backend = TorchBackend(device=self.device)
        #self.bound_backend = self.backend

    def _setup_buffers(self):
        N = min(self.X.shape[0], self.max_n)
        self.A = torch.zeros([N, N], device=self.device, requires_grad=False, dtype=self.y.dtype)
        self.K = None
        self.alpha0 = torch.zeros([N, 1], device=self.device, requires_grad=False, dtype=self.y.dtype)

    @classmethod
    def _get_kernel_evaluator(cls, X, k, sn2, K):
        def kernel_evaluator(A, i0, i1, j0, j1):
            if i0 == j0 and i1 == j1:
                # TODO: is there a better way to fill the diagonal? Allocating a whole identity matrix seems expensive
                K[i0:i0 + i1, i0:i0 + i1] = torch.tril(k(X[i0:i0 + i1, :]) + sn2 * torch.eye(i1, device=K.device, dtype=X.dtype))
                # copy values into designated array
                A[i0:i0 + i1, j0:j0 + j1] = K[i0:i0 + i1, j0:j0 + j1].clone()
            elif j1 <= i0:
                K[i0:i0 + i1, j0:j0 + j1] = k(X[i0:i0 + i1, :], X[j0:j0 + j1, :])
                # copy values into designated array
                A[i0:i0 + i1, j0:j0 + j1] = K[i0:i0 + i1, j0:j0 + j1].clone()
            else:
                raise RuntimeError("This case should not occur")

        return kernel_evaluator

    @torch.no_grad()
    def _make_err_copy(self, err):
        self.alpha0[:] = err[:]

    def _get_K(self):
        # TODO: Can we avoid reallocating this much memory?
        return torch.zeros_like(self.A) #, device=self.device)

    def _get_hook(self):
        # same call as super-class albeit not .item() following self.sn2()
        # torch is really bad at handling floats ...
        return StopHook(N=self.X.shape[0], min_noise=self.sn2(),
                        relative_tolerance=self.error_tolerance(), backend=self.backend)
