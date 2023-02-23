from math import pi

import mlflow
import torch
import gpytorch
from gpytorch import delazify
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from hyperparameter_tuning.utils.abstract_hyper_parameter_tuning_algorithm import AbstractHyperParameterTuningAlgorithm
from external.robustgp.inducing_input_init import ConditionalVariance


NUM_INDUCING_INPUTS = "NUM_INDUCING_INPUTS"
SELECTION_SCHEME = "SELECTION_SCHEME"
RANDOM = "RANDOM"
CONDITIONAL_VARIANCE = "CONDITIONAL_VARIANCE"
MAX_NUM_CG_STEPS = "MAX_NUM_CG_STEPS"
OPTIMIZE_INDUCING_INPUTS = "OPTIMIZE_INDUCING_INPUTS"


class NativeVariationalGPR(AbstractHyperParameterTuningAlgorithm):
    @staticmethod
    def add_parameters_to_parser(parser):
        pass
        #parser.add_argument("-ni", "--" + NUM_INDUCING_INPUTS, type=int, default=1024)
        #parser.add_argument("-ss", "--" + SELECTION_SCHEME, type=str, choices=[RANDOM, CONDITIONAL_VARIANCE],
        #                    default=CONDITIONAL_VARIANCE)
        ## the commented option below is not really supported. That's why we use the construction thereafter
        ##parser.add_argument("-oi", "--" + OPTIMIZE_INDUCING_INPUTS, type=bool, default=False)
        #parser.add_argument(f'--{OPTIMIZE_INDUCING_INPUTS}', dest=OPTIMIZE_INDUCING_INPUTS, action='store_true')
        #parser.add_argument(f'--no-{OPTIMIZE_INDUCING_INPUTS}', dest=OPTIMIZE_INDUCING_INPUTS, action='store_false')
        #parser.set_defaults(OPTIMIZE_INDUCING_INPUTS=True)

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: callable,
        mu: callable,
        args,
        device="cpu",
        **kwargs,
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)
        self.optimize_inducing_inputs = args[OPTIMIZE_INDUCING_INPUTS]
        mlflow.set_tag(OPTIMIZE_INDUCING_INPUTS, self.optimize_inducing_inputs)
        num_inducing_inputs = args[NUM_INDUCING_INPUTS]
        mlflow.set_tag(NUM_INDUCING_INPUTS, num_inducing_inputs)
        selection_scheme = args[SELECTION_SCHEME]
        mlflow.set_tag(SELECTION_SCHEME, selection_scheme)
        if selection_scheme == RANDOM:
            # we assume here that the dataset has been shuffled
            inducing_points = X[-num_inducing_inputs:, :]
        elif selection_scheme == CONDITIONAL_VARIANCE:
            wrapper = (
                lambda x1, x2, full_cov: gpytorch.delazify(k(x1, x2, diag=not full_cov))
                .detach()
                .cpu()
                .numpy()
            )
            inducing_points, indices = ConditionalVariance(
                X, num_inducing_inputs, wrapper
            )
        else:
            raise RuntimeError(f"Unknown selection scheme: {selection_scheme}")
        self.inducing_points = torch.nn.Parameter(inducing_points)
        self.const = X.shape[0] / 2 * torch.log(2 * torch.tensor(pi))

    def create_loss_closure(self):
        # The following code stems from https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py

        # Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
        #
        # Licensed under the Apache License, Version 2.0 (the "License");
        # you may not use this file except in compliance with the License.
        # You may obtain a copy of the License at
        #
        # http://www.apache.org/licenses/LICENSE-2.0
        #
        # Unless required by applicable law or agreed to in writing, software
        # distributed under the License is distributed on an "AS IS" BASIS,
        # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        # See the License for the specific language governing permissions and
        # limitations under the License.
        def loss():
            kernel = self.k
            inducing_variable = self.inducing_points

            sigma_sq = self.sn2()

            num_inducing = len(inducing_variable)
            sigma = torch.sqrt(sigma_sq)
            jitter = 1e-8  # this is the gpflow default
            jitter = torch.tensor(jitter, dtype=sigma.dtype, device=sigma.device)

            # kuf_lazy: GPytorchLazyTensor = _eval_kernel(kernel, inducing_variable, x_data)
            kuf_lazy = kernel(inducing_variable, self.X)
            kuf = delazify(kuf_lazy)

            # kuu: GPytorchLazyTensor = _eval_kernel(kernel, inducing_variable, inducing_variable)
            kuu = kernel(inducing_variable, inducing_variable)
            kuu_jitter = delazify(kuu.add_diag(jitter))
            kuu_chol = torch.cholesky(kuu_jitter)
            self.L = kuu_chol

            # Compute intermediate matrices
            trisolve = torch.triangular_solve
            A = trisolve(kuf, kuu_chol, upper=False).solution / sigma
            AAt = A @ A.transpose(-1, -2)
            I = torch.eye(num_inducing, dtype=AAt.dtype, device=AAt.device)
            B = AAt + I
            LB = torch.cholesky(B)
            self.LB = LB
            #AAt_diag_sum = AAt.diagonal().sum()

            kdiag = kernel(self.X, diag=True)

            # tr(K) / σ²
            trace_k = torch.sum(kdiag) / sigma_sq
            # tr(Q) / σ²
            trace_q = torch.sum(AAt.diagonal())
            # tr(K - Q) / σ²
            trace = trace_k - trace_q

            # 0.5 * log(det(B))
            half_logdet_b = torch.sum(torch.log(LB.diagonal()))

            # N * log(σ²)
            log_sigma_sq = self.X.shape[0] * torch.log(sigma_sq)

            logdet_k = -(half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)

            err = self.y - self.mu(self.X)

            Aerr = A @ err
            c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma
            self.c = c

            # σ⁻² yᵀy
            err_inner_prod = torch.sum(torch.square(err)) / sigma_sq
            c_inner_prod = torch.sum(torch.square(c))

            quad = -0.5 * (err_inner_prod - c_inner_prod)
            return -quad - logdet_k + self.const  # signs are inverted

        return loss

    def get_posterior(self, X_test, full_posterior=False):
        if full_posterior:
            raise NotImplementedError()
        trisolve = torch.triangular_solve
        kus = self.k(self.inducing_points, X_test)
        tmp1 = trisolve(delazify(kus), self.L, upper=False)[0]
        tmp2 = trisolve(tmp1, self.LB, upper=False)[0]

        sgpr_mean = tmp2.transpose(-1, -2) @ self.c + self.mu(X_test)

        kss = self.k(X_test, diag=True)
        f_var = delazify(kss) + (tmp2.square()).sum(0) - (tmp1.square()).sum(0)
        f_var = f_var.reshape(*sgpr_mean.shape)
        return sgpr_mean, f_var

    def requires_ground_truth_recording(self):
        return True

    def get_named_tunable_parameters(self):
        ls = []
        if self.optimize_inducing_inputs:
            ls.append(("inducing_points", self.inducing_points))
        return ls


class _GPyTorchVariationalGPR(gpytorch.models.ApproximateGP):
    def __init__(self, mean, kernel, inducing_points, learn_inducing_locations=True):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        self.mean_module = mean
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPR(NativeVariationalGPR):
    @staticmethod
    def add_parameters_to_parser(parser):
        pass
        # we rely here that the super class has already added the parameters

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: callable,
        mu: callable,
        args,
        device="cpu",
        **kwargs,
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)
        self._initialize_model(self.inducing_points)

    def _initialize_model(self, inducing_points):
        mean_module = gpytorch.means.constant_mean.ConstantMean()
        mean_module.initialize(constant=self.mu(None))
        mean_module.train()

        self.model = _GPyTorchVariationalGPR(
            mean_module,
            self.k,
            inducing_points,
            learn_inducing_locations=self.optimize_inducing_inputs  # this is handled separately  # TODO: or should this be set True?
        )
        self.model.train()

        sn2 = self.sn2
        class NoiseHook(HomoskedasticNoise):
            @property
            def noise(self):
                return sn2().reshape(1)

            @noise.setter
            def noise(self, value) -> None:
                raise NotImplementedError("This should not be called!")

        self.likelihood = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood()
        self.likelihood.noise_covar = NoiseHook()
        self.likelihood.train()

    def create_loss_closure(self):
        def loss():
            self.model.mean_module.constant = self.mu(None)
            output = self.model(self.X)
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.y.size(0))
            return -mll(output, self.y.squeeze()) * self.X.shape[0]

        return loss

    def get_posterior(self, X_test, full_posterior=False):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test))

        if full_posterior:
            ret = observed_pred.mean[:, None], observed_pred.covariance_matrix
        else:
            ret = observed_pred.mean[:, None], observed_pred.variance[:, None]
        self.model.train()
        self.likelihood.train()
        return ret

    def requires_ground_truth_recording(self):
        return True

    def get_named_tunable_parameters(self):
        ls = []
        if self.optimize_inducing_inputs:
            inducing_inputs = self.model.variational_strategy.inducing_points
            ls.append(("model.variational_strategy.inducing_points", inducing_inputs))
        return ls


class GPyTorchVariationalGPR(AbstractHyperParameterTuningAlgorithm):
    "Sparse variational GP model using GPyTorch to the largest extent possible."

    @staticmethod
    def add_parameters_to_parser(parser):
        parser.add_argument("-ni", "--" + NUM_INDUCING_INPUTS, type=int, default=2048)
        parser.add_argument("-ss", "--" + SELECTION_SCHEME, type=str, choices=[RANDOM, CONDITIONAL_VARIANCE],
                            default=CONDITIONAL_VARIANCE)
        # the commented option below is not really supported. That's why we use the construction thereafter
        #parser.add_argument("-oi", "--" + OPTIMIZE_INDUCING_INPUTS, type=bool, default=False)
        parser.add_argument(f'--{OPTIMIZE_INDUCING_INPUTS}', dest=OPTIMIZE_INDUCING_INPUTS, action='store_true')
        parser.add_argument(f'--no-{OPTIMIZE_INDUCING_INPUTS}', dest=OPTIMIZE_INDUCING_INPUTS, action='store_false')
        parser.set_defaults(OPTIMIZE_INDUCING_INPUTS=True)

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k: gpytorch.kernels.Kernel,
        sn2: gpytorch.likelihoods.Likelihood,
        mu: gpytorch.means.Mean,
        args,
        device="cpu",
        **kwargs,
    ):
        super().__init__(X, y, k, sn2, mu, args, device=device)

        num_inducing_inputs = args[NUM_INDUCING_INPUTS]
        selection_scheme = args[SELECTION_SCHEME]
        if selection_scheme == RANDOM:
            # we assume here that the dataset has been shuffled
            inducing_points = X[-num_inducing_inputs:, :]
        elif selection_scheme == CONDITIONAL_VARIANCE:
            wrapper = (
                lambda x1, x2, full_cov: gpytorch.delazify(k(x1, x2, diag=not full_cov))
                .detach()
                .cpu()
                .numpy()
            )
            inducing_points, indices = ConditionalVariance(
                X, num_inducing_inputs, wrapper
            )
        else:
            raise RuntimeError(f"Unknown selection scheme: {selection_scheme}")

        self.likelihood = sn2
        self.model = _GPyTorchVariationalGPR(
            mu,
            k,
            inducing_points,
            learn_inducing_locations=args[OPTIMIZE_INDUCING_INPUTS],
        )

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=y.size(0)
        )

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
        for parameter_group in [self.model, self.likelihood]:
            for n, p in parameter_group.named_parameters():
                yield n, p

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
        return True
