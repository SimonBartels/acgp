import mlflow
import torch
import gpytorch
import numpy as np

from external.robustgp.inducing_input_init import ConditionalVariance
from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import StoppedCholesky


PRECONDITIONER_STEPS = "PRECONDITIONER_STEPS"


class PreconditionedStoppedCholesky(StoppedCholesky):
    @staticmethod
    def add_parameters_to_parser(parser):
        parser.add_argument("-ps", f"--{PRECONDITIONER_STEPS}", type=int, default=2048)

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: callable,
        mu: callable,
        args,
        device="cpu"
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)
        M = args[PRECONDITIONER_STEPS]
        mlflow.set_tag(PRECONDITIONER_STEPS, M)

        wrapper = (
            lambda x1, x2, full_cov: gpytorch.delazify(k(x1, x2, diag=not full_cov))
            .detach()
            .cpu()
            .numpy()
        )
        _, indices = ConditionalVariance(
            X, M, wrapper
        )
        N = X.shape[0]
        perm = np.arange(N)
        perm[:M] = indices
        perm[M:] = np.setdiff1d(np.arange(N), indices)
        # TODO: the following copy can be avoided, but it shouldn't be too bad
        self.X = self.X[perm, :]
        self.y = self.y[perm, :]
