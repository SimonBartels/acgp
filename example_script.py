from math import pi
import torch
from gpytorch.kernels import RBFKernel

from acgp.blas_wrappers.torch_wrapper import TorchWrapper
from acgp.hooks.stop_hook import StopHook
from acgp.meta_cholesky import MetaCholesky
from acgp.backends.torch_backend import TorchBackend

# computing gradients is currently a bit tricky: see stopped_choleksy in the hyper_parameter_tuning package
with torch.no_grad():
    N = 5000  # number of data points
    Nmax = N  # maximal amount of memory we are willing to spend
    D = 2  # dimensionality of the data
    X = torch.randn([N, D], dtype=torch.float64)
    k = RBFKernel()
    sn2 = torch.as_tensor(1e-3, dtype=X.dtype)  # observation noise
    L = torch.linalg.cholesky(k(X).evaluate() + sn2 * torch.eye(N, dtype=X.dtype))  # exact Cholesky
    # when the targets are indeed samples from the GP, mean prediction is easy and stopping should be possible early
    y = L @ torch.randn([N, 1], dtype=X.dtype)
    #y = torch.randn([N, 1], dtype=X.dtype)  # hard case, stopping will occur late

    r = 1e-1  # desired relative error
    block_size = 100  # number of data points that is processed in batches and number of samples to estimate the bounds
    cholesky_algorithm = MetaCholesky(block_size=block_size, initial_block_size=block_size, blaswrapper=TorchWrapper())

    # wrapper for the kernel function to fill the kernel matrix
    def kernel_evaluator(K, i0, i1, j0, j1):
        if i0 == j0 and i1 == j1:
            K[i0:i0 + i1, i0:i0 + i1] = k(X[i0:i0 + i1, :]).evaluate() + sn2 * torch.eye(i1, device=K.device, dtype=X.dtype)
        elif j1 <= i0:
            K[i0:i0 + i1, j0:j0 + j1] = k(X[i0:i0 + i1, :], X[j0:j0 + j1, :]).evaluate()
        else:
            raise RuntimeError("This case should not occur")


    K = torch.empty([Nmax, Nmax], dtype=torch.float64)  # preallocate memory
    alpha = y.clone()  # this will contain L^-1 y
    hook = StopHook(N=N, min_noise=sn2, relative_tolerance=r, backend=TorchBackend())  # create hook which actually monitors the bounds
    cholesky_algorithm.run_configuration(K, err=alpha, kernel_evaluator=kernel_evaluator, hook=hook)  # run Cholesky with hook

    # collect result
    llh_approximation, bound_history = hook.get_bounds()
    M = hook.iteration  # size of the processed subset
    print(f"{M} of {N} data points have been fully processed.")
    Udet, Ldet, Uquad, Lquad = bound_history[-1]  # bounds on determinant and quadratic form

    # compute ground-truth
    logdet = 2 * torch.sum(torch.log(torch.diag(L)))  # log-determinant
    print(f"Absolute difference to true log-determinant: {Udet / 2 + Ldet / 2 - logdet}")
    alpha, _ = torch.triangular_solve(y, L, upper=False)  # temporary vector
    yKy = torch.sum(torch.square(alpha.squeeze()))  # solution to the quadratic form
    print(f"Absolute difference to true quadratic form: {Uquad / 2 + Lquad / 2 - yKy}")
    llh = logdet / 2 + yKy / 2 + N * torch.log(torch.as_tensor(2. * pi, dtype=X.dtype)) / 2
    print(f"Relative approximation error on marginal log-likelihood {torch.abs((llh_approximation - llh) / llh)}. "
          f"Desired precision was {r}.")
    print(f"The block-size should be on the order of 10K points for reliable estimation.")
