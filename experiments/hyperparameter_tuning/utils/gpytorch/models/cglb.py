import gpytorch
import mlflow
import torch
from gpytorch.kernels import InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood

from external.cglb.backend.pytorch.models import CGLB as CGLB_, LowerBoundCG, PredictCG
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import VariationalGPR

MAX_NUM_CG_STEPS = "MAX_NUM_CG_STEPS"


class CGLB(VariationalGPR):
    @staticmethod
    def add_parameters_to_parser(parser):
        # We rely here that VariationalGPR has already added the parameters to the parser.
        #super(CGLB, CGLB).add_parameters_to_parser(parser)
        parser.add_argument("-cgs", "--" + MAX_NUM_CG_STEPS, type=int, default=100)

    def __init__(self, X: torch.Tensor, y: torch.Tensor, k: gpytorch.kernels.Kernel,
                 sn2: callable, mu: callable, args, device="cpu", **kwargs):
        self.max_cg_iter = args[MAX_NUM_CG_STEPS]
        mlflow.set_tag(MAX_NUM_CG_STEPS, self.max_cg_iter)
        super().__init__(X, y, k, sn2, mu, args, device=device)

    def _initialize_model(self, inducing_points):
        likelihood = GaussianLikelihood().to(self.device)
        likelihood.train()

        kernel = InducingPointKernel(base_kernel=self.k, inducing_points=inducing_points, likelihood=likelihood)
        kernel.train()
        data = (self.X, self.y)
        self.cglbmodel = CGLB_(data=data, kernel=kernel, likelihood=likelihood).to(self.device)
        self.cglbmodel.train()

        sn2 = self.sn2  # no parenthesis!
        mu = self.mu
        max_cg_iter = self.max_cg_iter

        class _PredictCG(PredictCG):
            @property
            def noise(self) -> torch.Tensor:
                return sn2()

            @property
            def mean(self):
                return mu  # CGLB expects a callable, no parenthesis!

            def loss(self):
                return -LowerBoundCG.forward(self, data, max_cg_iter=max_cg_iter)
        self.cglb_post = _PredictCG(model=self.cglbmodel)

    def log_metrics(self, step: int):
        # the v cache is cleared in the callback step
        # this call occurs during an accepted step of the optimizer
        self.cglb_post.clear_cache()
        #self.cglb_post.cached_v_vec = False

    def create_loss_closure(self):
        return self.cglb_post.loss

    def get_posterior(self, X_test, full_posterior=False):
        self.cglbmodel.train(False)
        self.cglbmodel.likelihood.train(False)
        ret = self.cglb_post(X_test, full_cov=full_posterior)
        self.cglbmodel.train()
        self.cglbmodel.likelihood.train()
        return ret

    def get_named_tunable_parameters(self):
        ls = []
        if self.optimize_inducing_inputs:
            inducing_inputs = self.cglb_post.model.covar_module.inducing_points
            ls.append(("model.covar_module.inducing_points", inducing_inputs))
        return ls
