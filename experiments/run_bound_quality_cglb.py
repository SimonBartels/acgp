"""
Computes bounds for the same experiment setup as in #run_ground_truth_experiments.py
"""
import torch
import numpy as np
import random
import gc
import mlflow
import numpy.random
from gpytorch.lazy import DiagLazyTensor, NonLazyTensor
from gpytorch.likelihoods import GaussianLikelihood
from time import thread_time, time

from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import NUM_INDUCING_INPUTS
from external.cglb.backend.pytorch.models import LowerBoundCG
from external.cglb.backend.pytorch.models import CGLB as CGLBmodel
from hyperparameter_tuning.utils.gpytorch.models.cglb import MAX_NUM_CG_STEPS
from bound_quality.parser import get_parser
from external.robustgp.inducing_input_init import ConditionalVariance
from utils.data.load_dataset import load_dataset
from utils.registry import KERNEL_DICT, ENVIRONMENT_DICT
from utils.execution.run_cluster import execute_job_array_on_slurm_cluster
from utils.result_management.constants import SETUP_TIME, STEP_TIME, L_DET, U_DET, L_QUAD, U_QUAD, EXTRA_TIME, \
    SETUP_TIME_WC, STEP_TIME_WC, ENV_CPUS
from utils.result_management.result_management import define_run, set_experiment_tags, \
    initialize_experiment, get_bound_results_path, make_experiment_name

mlflow.set_tracking_uri(get_bound_results_path())


def main(**args):
    if args["mode"] == "execute_single_batch_job":
        execute_single_batch_job(**args)
        #execute_post_single_batch_job(**args)
    elif args["mode"] == "generate_batch_jobs":
        generate_batch_jobs(**args)
        #generate_post_experiment_batch_jobs(**args)
    else:
        raise RuntimeError(f"unknown mode: {args['mode']}")


@torch.no_grad()
def execute_single_batch_job(**args):
    dtype = torch.float64
    seed = args["seed"]
    dataset = args["dataset"]
    sn2 = args["sn2"]

    experiment_name = args["experiment_name"]
    if experiment_name is None:
        raise RuntimeError("The experiment needs to have a name!")

    mlflow.set_experiment(experiment_name)

    # set seeds
    np.random.seed(seed)
    random.seed(seed)

    # prepare variables
    X, y = load_dataset(dataset)
    N = X.shape[0]
    p = numpy.random.permutation(N)
    X = X[p, :]
    y = y[p, :]

    k = KERNEL_DICT[args["kernel"]].initialize_from_parser(args)

    def gpflow_kernel_wrapper(x1, x2=None, full_cov=False):
        if not full_cov:
            return k.Kdiag(x1)
        return k.K(x1, x2)

    parameters = {}
    num_inducing_inputs = args[NUM_INDUCING_INPUTS]
    #mlflow.set_tag(NUM_INDUCING_INPUTS, num_inducing_inputs)
    parameters[NUM_INDUCING_INPUTS] = num_inducing_inputs
    max_cg_iter = args[MAX_NUM_CG_STEPS]
    #mlflow.set_tag(MAX_NUM_CG_STEPS, max_cg_iter)
    parameters[MAX_NUM_CG_STEPS] = max_cg_iter

    mlflow.start_run()
    t0_wc = time()
    t0 = thread_time()
    Z, indices = ConditionalVariance(X, num_inducing_inputs, gpflow_kernel_wrapper)
    mlflow.log_metric(SETUP_TIME, thread_time() - t0)
    mlflow.log_metric(SETUP_TIME_WC, time() - t0_wc)

    likelihood = GaussianLikelihood()
    likelihood.train()

    def gpytorch_kernel_wrapper(self, x1, x2=None, diag=False):
        if diag:
            return DiagLazyTensor(torch.tensor(k.Kdiag(x1.numpy())))
        if x2 is None:
            return NonLazyTensor(torch.tensor(k.K(x1.numpy())))
        return NonLazyTensor(torch.tensor(k.K(x1.numpy(), x2.numpy())))

    class WrapperKernel:
        base_kernel = gpytorch_kernel_wrapper
        inducing_points = torch.tensor(Z)

    kernel = WrapperKernel()
    data = (torch.tensor(X), torch.tensor(y))
    cglbmodel = CGLBmodel(data=data, kernel=kernel, likelihood=likelihood)
    cglbmodel.train()

    noise = torch.tensor(sn2, dtype=dtype)
    class CGLB(LowerBoundCG):
        @property
        def noise(self) -> torch.Tensor:
            return noise

        def get_signature(self):
            return "cglb" + str(num_inducing_inputs)

        def mean(self, x):
            return torch.zeros(x.shape[0], dtype=dtype)

    cglb = CGLB(model=cglbmodel)

    define_run(cglb, parameters, seed)
    # we must have defined the run first, to have an active run object...
    set_experiment_tags(dataset, k, ENVIRONMENT_DICT, sn2)

    gc.collect()
    gc.disable()

    t0_wc = time()
    t0 = thread_time()
    common_terms = cglb.logdet_and_quad_common_terms(data)
    quad_bounds = cglb.quad_estimator(data, terms=common_terms, max_cg_iter=max_cg_iter)
    """
    NOTE: cglb returns the negative!
    """
    udet_bound = -cglb.logdet_estimator(data, terms=common_terms).item()
    lb = 2 * common_terms.LB.diagonal().log().sum().item() + N * np.log(sn2)
    mlflow.log_metric(STEP_TIME, thread_time() - t0, step=0)
    mlflow.log_metric(STEP_TIME_WC, time() - t0_wc)
    mlflow.log_metric(U_DET, 2 * udet_bound, step=0)
    mlflow.log_metric(U_QUAD, -2 * quad_bounds.upper_bound.item(), step=0)
    mlflow.log_metric(L_QUAD, -2 * quad_bounds.lower_bound.item(), step=0)

    gc.collect()
    gc.disable()

    # TODO: Implement lower bound
    t0 = thread_time()
    # A = common_terms.A
    # Kuu = (A.T @ A).detach().numpy()
    # Kuudiag = np.diag(Kuu).copy()
    # np.fill_diagonal(Kuu, 0.)
    # radii = np.squeeze(np.sum(np.abs(Kuu), axis=0))
    # l1 = np.max(Kuudiag + radii) + sn2
    # # log_trace = log(1 + (trace_kff - trace_qrest) / l1)
    # #trace_diff = np.sum(1. - Kuudiag)
    # #log_trace = np.log(trace_diff) - np.log(N) - np.log(l1)
    # #lb = log_det_q + log_trace
    # lb = udet_bound.item() + np.log(sn2) - np.log(l1)
    # lb = 2 * lb

    mlflow.log_metric(EXTRA_TIME, thread_time() - t0, step=0)
    mlflow.log_metric(L_DET, lb, step=0)
    mlflow.end_run()


def generate_batch_jobs(**args):
    cpus = 40

    seeds = range(0, 5)
    datasets = ['metro', 'pm25', 'protein', 'wilson_kin40k']
    ls = [-1., 0., 1., 2.]

    ks = list(KERNEL_DICT.keys())

    num_inducing_inputs = [512, 1024, 2048, 4096]

    sn2 = 1e-3
    theta = 0.

    template = "python %s -m execute_single_batch_job" % __file__
    commands = []
    for seed in seeds:
        for ni in num_inducing_inputs:
            for l in ls:
                for k in ks:
                    kernel = KERNEL_DICT[k].initialize(log_var=theta, log_ls2=l)
                    for dataset in datasets:
                        # make sure the experiment exists to avoid conflicts
                        #experiment_name = initialize_experiment(dataset=dataset, kernel=kernel, sn2=sn2, cpus=cpus)
                        experiment_name = make_experiment_name(dataset, kernel, {ENV_CPUS: cpus}, sn2)
                        command = template + " -en %s -d %s" % (experiment_name, dataset)
                        command += " -k " + kernel.name + kernel.generate_command_string()
                        command += " -sn2 %f" % sn2
                        command += " -s %i" % seed
                        command += f" -ni {ni}"
                        commands.append(command)
                        #cluster_command = execute_single_configuration_on_slurm_cluster(command=command, cpus=cpus)
                        #print("executing: %s" % cluster_command)
                        #print("with: %s" % command)
                        #run_local(command)
    execute_job_array_on_slurm_cluster(commands, cpus=cpus, mem=50000)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("-ni", "--" + NUM_INDUCING_INPUTS, type=int, default=512)
    parser.add_argument("-cgs", "--" + MAX_NUM_CG_STEPS, type=int, default=100)

    args = parser.parse_args()
    main(**vars(args))
