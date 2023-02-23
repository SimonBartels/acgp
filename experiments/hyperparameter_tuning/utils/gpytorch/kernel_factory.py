import gpytorch
import mlflow
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, MaternKernel, PolynomialKernel, ScaleKernel, RQKernel
from gpytorch.kernels import InducingPointKernel as InducingPointKernel_
from gpytorch.lazy import delazify
from gpytorch.utils.cholesky import psd_safe_cholesky

from external.robustgp.inducing_input_init import ConditionalVariance


def _raise(e):
    raise e


kernel_dict = {
    "RBF": lambda D, ls: ScaleKernel(RBFKernel()),
    "ARDSE": lambda D, ls: ScaleKernel(RBFKernel(ard_num_dims=D)),
    "Matern32": lambda D, ls: ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=D)),
    "Matern52*Cubic": lambda D, ls: ScaleKernel(MaternKernel(ard_num_dims=D)) * PolynomialKernel(power=3),
    "ArtemevExp": lambda D, *args: ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=D)),
    "MaternInducing": lambda *args: _raise(RuntimeError("can not instantiate this kernel like this")),
    #"InducingMatern": lambda *args: _raise(RuntimeError("can not instantiate this kernel like this")),
    "InducingPlus": lambda *args: _raise(RuntimeError("can not instantiate this kernel like this"))
}

KERNEL_NAME = "kernel_name"
LENGTH_SCALE = "length_scale"


class KernelFactory:
    def add_parameters_to_parser(self, parser):
        parser.add_argument("-ls", "--" + LENGTH_SCALE, type=float, default=1.0)

    def create(self, args, X):
        D = X.shape[1]
        kernel_name = args[KERNEL_NAME]
        length_scale = args[LENGTH_SCALE]
        if kernel_name == "MaternInducing":
            lk = None #gpytorch.likelihoods.GaussianLikelihood()
            k_ = MaternKernel(nu=1.5, ard_num_dims=D)
            k_.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-6))
            wrapper = lambda x1, x2, full_cov: delazify(k_(x1, x2, diag=not full_cov)).detach().numpy()
            Z, indices = ConditionalVariance(X, 1024, wrapper)
            k = ScaleKernel(InducingPointKernel(k_, Z, lk))
        elif kernel_name == "InducingPlus":
            Z = X[-1024:, :].clone()
            lk = None #gpytorch.likelihoods.GaussianLikelihood()
            k_ = MaternKernel(nu=1.5, ard_num_dims=D)
            k_.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-6))
            k1 = InducingPointKernel(k_, Z, lk)
            k2 = RQKernel(alpha_constraint=gpytorch.constraints.GreaterThan(1e-4))
            k2.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-6))
            k = ScaleKernel(k1) + ScaleKernel(k2)
        else:
            k: gpytorch.kernels.Kernel = kernel_dict[kernel_name](D, length_scale)
            if kernel_name == "ArtemevExp":
                # even though we register the constraint for the raw parameter, it actually applies to the final parameter
                k.base_kernel.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(1e-6))
        return k

    def add_tags_to_experiment(self, args):
        experiment_id = mlflow.active_run().info.experiment_id
        kernel_name = args[KERNEL_NAME]
        mlflow.tracking.MlflowClient().set_experiment_tag(experiment_id, KERNEL_NAME, kernel_name)

    def get_available_kernel_functions(self):
        return list(kernel_dict.keys())



class InducingPointKernel(InducingPointKernel_):
    def forward(self, x1, x2, diag=False, **kwargs):
        covar = self._get_covariance(x1, x2)
        if diag:
            return covar.diag()
        else:
            return covar

    def _get_covariance(self, x1, x2):
        k_ux1 = delazify(self.base_kernel(x1, self.inducing_points)).T
        chol = self._inducing_chol
        k_ux1 = torch.triangular_solve(k_ux1, chol, upper=False).solution.T
        if torch.equal(x1, x2):
            k_ux2 = k_ux1
        else:
            k_ux2 = delazify(self.base_kernel(x2, self.inducing_points)).T
            k_ux2 = torch.triangular_solve(k_ux2, chol, upper=False).solution.T

        return k_ux1 @ k_ux2.T

    def _clear_cache(self):
        super()._clear_cache()
        if hasattr(self, "_cached_inducing_chol"):
            del self._cached_inducing_chol


    @property
    def _inducing_chol(self):
        if not self.training and hasattr(self, "_cached_inducing_chol"):
            return self._cached_inducing_chol
        else:
            chol = psd_safe_cholesky(self._inducing_mat, upper=False)
            if not self.training:
                self._cached_inducing_chol = chol
            return chol
