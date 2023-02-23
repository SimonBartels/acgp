import math

import torch
import gpytorch
import gpytorch.kernels
import torch.linalg

from hyperparameter_tuning.utils.abstract_hyper_parameter_tuning_algorithm import (
    AbstractHyperParameterTuningAlgorithm,
)
from acgp.models.custom_reverse.pytorch.log_marginal import get_custom_log_det_plus_quad


class ExactGPR(AbstractHyperParameterTuningAlgorithm):
    @staticmethod
    def add_parameters_to_parser(parser):
        pass

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: callable,
        mu: callable,
        args,
        device="cpu",
        **kwargs
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)
        self.last_loss = None
        N = X.shape[0]
        self.L = torch.zeros([N, N], dtype=X.dtype, device=device)
        self.alpha = torch.zeros([N, 1], dtype=X.dtype, device=device)
        self.ldet_plus_quad = torch.zeros(1, dtype=X.dtype, device=device)

        # For adding jitter to the covariance matrix diagonal:
        self.I = torch.eye(X.shape[0], device=self.device, dtype=X.dtype)

    def create_loss_closure(self):
        const = (
            self.X.shape[0]
            / 2
            * torch.log(
                torch.tensor(2 * math.pi, requires_grad=False, device=self.device)
            )
        )
        custom = get_custom_log_det_plus_quad(
            self.L, self.alpha, self.ldet_plus_quad, subset_size=[self.X.shape[0]]
        )

        def loss():
            K = self.k(self.X).evaluate() + self.sn2() * self.I
            err = self.y - self.mu(self.X)
            with torch.no_grad():
                success = False
                jitter = 1e-8
                while not success:
                    try:
                        torch.linalg.cholesky(K + jitter * self.I, out=self.L)
                    except RuntimeError:
                        jitter *= 10
                    else:
                        success = True
                L = self.L
                # alpha, _ = torch.triangular_solve(err, L, upper=False)
                ldet = torch.sum(torch.log(torch.diag(L))) * 2
                # quad = torch.sum(torch.square(alpha))
                # # WHY ON EARTH DOES TORCH HAVE TO COPY THE COEFFICIENT MATRIX???
                # torch.triangular_solve(alpha, L.T, out=(self.alpha, L))
                torch.cholesky_solve(err, L, out=self.alpha)
                quad = err.T @ self.alpha
                self.ldet_plus_quad[0] = ldet + quad
            # the following function call makes the gradient calculation way more efficient
            ldet_plus_quad = custom.apply(K, err)
            self.last_loss = ldet_plus_quad / 2 + const
            return self.last_loss

        return loss

    def get_posterior(self, X_test, full_posterior=False):
        with torch.no_grad():
            beta = self.k(self.X, X_test).evaluate()
            mu = self.mu(X_test) + beta.T @ self.alpha
            torch.triangular_solve(beta, self.L, upper=False, out=(beta, self.L))
            if not full_posterior:
                var = self.k(X_test, diag=True) - torch.sum(torch.square(beta), dim=[0])
                var = torch.reshape(var, (-1, 1))
            else:
                var = self.k(X_test).evaluate() - beta.T @ beta
            return mu, var

    def requires_ground_truth_recording(self):
        return False


class _GPyTorchExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, mean, kernel, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPyTorchExactGPR(AbstractHyperParameterTuningAlgorithm):
    "Exact GP model using GPyTorch to the largest extent possible."

    @staticmethod
    def add_parameters_to_parser(parser):
        pass

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: gpytorch.likelihoods.Likelihood,
        mu: gpytorch.means.Mean,
        args,
        device="cpu",
        **kwargs
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)

        self.likelihood = sn2
        self.model = _GPyTorchExactGPR(X, y, mu, k, self.likelihood)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.last_loss = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def create_loss_closure(self):
        def loss():
            self.model.train()
            self.likelihood.train()

            output = self.model(self.X)
            loss = -self.mll(output, self.y)

            self.last_loss = loss

            return self.last_loss

        return loss

    def named_parameters(self):
        return self.model.named_parameters()

    def get_posterior(self, X_test, full_posterior=False):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test))

        if full_posterior:
            return observed_pred.mean[:, None], observed_pred.covariance_matrix
        else:
            return observed_pred.mean[:, None], observed_pred.variance[:, None]

    def requires_ground_truth_recording(self):
        return False
