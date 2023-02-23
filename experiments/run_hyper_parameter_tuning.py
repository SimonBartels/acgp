"""
Script to run hyper-parameter tuning experiments.
"""
import gc
import itertools
import argparse
import logging
from gpytorch.utils.transforms import inv_softplus
from logging import log, INFO, DEBUG, WARNING, ERROR
import torch
import numpy as np
import random
import numpy.random
import mlflow
from torch.nn import Parameter
from warnings import warn

import utils.registry as registry
from hyperparameter_tuning.utils.gpytorch.models.cglb import CGLB
from hyperparameter_tuning.utils.gpytorch.models.exact_gpr import ExactGPR, GPyTorchExactGPR
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import GPyTorchVariationalGPR
from hyperparameter_tuning.utils.gpytorch.kernel_factory import KernelFactory, KERNEL_NAME
from hyperparameter_tuning.utils.gpytorch.recording import make_recording_callback
from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import StoppedCholesky
from hyperparameter_tuning.utils.gpytorch.optimizer import Scipy
from utils.data.load_dataset import get_train_test_dataset
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import VariationalGPR, NativeVariationalGPR, \
    OPTIMIZE_INDUCING_INPUTS
from utils.execution.run_cluster import execute_job_array_on_slurm_cluster, \
    execute_job_array_on_lsf_cluster
from utils.result_management.constants import ALGORITHM, SEED, DATASET, ALGORITHM_NAME, EXPERIMENT_TYPE, \
    HYPER_PARAMETER_TUNING_EXPERIMENT, PARSER_ARGS, OPTIMIZATION_MSG, SETUP_TIME, ENV_CPUS
from utils.result_management.result_management import get_results_path

try:
    from time import thread_time as time
    from time import process_time as process_time
except:
    warn("Failed to import thread_time! Going to use default time which might give different results.")
    from time import time
    process_time = lambda: 0

torch.set_default_dtype(torch.float64)

assert(mlflow.get_tracking_uri() == get_results_path())

HYPER_PARAMETER_TUNING_ALGOS = {
    ExactGPR.get_registry_key(): ExactGPR,
    GPyTorchExactGPR.get_registry_key(): GPyTorchExactGPR,
    GPyTorchVariationalGPR.get_registry_key(): GPyTorchVariationalGPR,
    StoppedCholesky.get_registry_key(): StoppedCholesky,
    CGLB.get_registry_key(): CGLB,
    VariationalGPR.get_registry_key(): VariationalGPR,
    NativeVariationalGPR.get_registry_key(): NativeVariationalGPR
}

MAX_ITERATIONS = "max_iterations"
MAX_TIME = "max_time"
OPTIMIZER = "optimizer"
LOWER_NOISE_CONSTRAINT = "lower_noise_constraint"
DEVICE = "device"

# The block size is openBLAS_block_size * number_of_cores we are going to use.
# For GPU experiments such a block_size would be larger than the dataset sizes, so we just use the same.
INITIAL_BLOCK_SIZE = 40 * 256
BLOCK_SIZE = INITIAL_BLOCK_SIZE


def main(**args):
    if args["mode"] == "execute_single_batch_job":
        execute_single_batch_job(**args)
    elif args["mode"] == "generate_batch_jobs":
        generate_batch_jobs(**args)
    elif args["mode"] == "generate_large_scale_batch_jobs":
        generate_large_scale_batch_jobs(**args)
    else:
        raise RuntimeError(f"unknown mode: {args['mode']}")


def execute_single_batch_job(**args):
    #torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(level=args["debug_level"])

    experiment_name = args["experiment_name"]
    if experiment_name is None:
        raise RuntimeError("The experiment needs to have a name!")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_dict(args, PARSER_ARGS)
    # we must have defined the run first, to have an active run object...
    experiment_id = mlflow.active_run().info.experiment_id
    dataset = args[DATASET]
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    mlfc = mlflow.tracking.MlflowClient()
    mlfc.set_experiment_tag(experiment_id, DATASET, dataset)
    mlfc.set_experiment_tag(experiment_id, EXPERIMENT_TYPE, HYPER_PARAMETER_TUNING_EXPERIMENT)
    # TODO: the experiment tags are supposed to be the same for all runs--how to assure that?

    max_iterations = args[MAX_ITERATIONS]
    mlflow.set_tag(MAX_ITERATIONS, str(max_iterations))
    max_time = args[MAX_TIME]
    mlflow.set_tag(MAX_TIME, str(max_time))
    seed = args["seed"]
    mlflow.set_tag(SEED, seed)  # the seed is a run property
    optimizer = args[OPTIMIZER]
    mlflow.set_tag(OPTIMIZER, optimizer)
    mlflow.set_tag(ENV_CPUS, registry.ENVIRONMENT_DICT[ENV_CPUS])

    # set seeds
    np.random.seed(seed)
    random.seed(seed)

    # prepare variables
    X, y, X_test, y_test = get_train_test_dataset(dataset, seed=seed)
    log(level=INFO, msg="dataset loaded")
    X = torch.as_tensor(X, dtype=dtype)
    y = torch.as_tensor(y, dtype=dtype)
    X_test = torch.as_tensor(X_test, dtype=dtype)
    y_test = torch.as_tensor(y_test, dtype=dtype)

    kf = KernelFactory()
    kf.add_tags_to_experiment(args)
    k = kf.create(args, X)
    if dtype == torch.float64:
        k.double()
    elif dtype == torch.float32:
        k.float()
    else:
        raise RuntimeError(f"unknown dtype: {dtype}")
    k.train(True)

    device = args[DEVICE]
    mlflow.set_tag(DEVICE, device)
    if device == "cuda":
        X = X.cuda()
        y = y.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        k = k.cuda()
        gc.collect()
    elif device == "cpu_small":
        device = "cpu"

    raw_sn2, sn2 = _instantiate_noise(args, device, dtype=dtype)
    raw_mu, mu = _instantiate_mean(args, device, dtype=dtype)

    if args[ALGORITHM] in ("GPyTorchExactGPR", "GPyTorchVariationalGPR"):
        import gpytorch
        likelihood = sn2 = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = args["initial_noise"]

        mu = gpytorch.means.ConstantMean()

        # GPyTorch crashes for the full dataset for some weird reason:
        if args[DATASET] == "toydata1000":
            X = X[:800]
            y = y[:800].squeeze()

        # GPyTorch expects scalar outputs:
        y = y.squeeze()

    mlflow.set_tag(ALGORITHM, args[ALGORITHM])
    t0 = time()
    algorithm = HYPER_PARAMETER_TUNING_ALGOS[args[ALGORITHM]](X, y, k, sn2, mu, args, device)
    closure = algorithm.create_loss_closure()
    t = time() - t0
    mlflow.set_tag(ALGORITHM_NAME, algorithm.get_name())
    mlflow.log_metric(SETUP_TIME, t, step=0)

    record_metrics_callback = make_recording_callback(algorithm=algorithm, X=X, y=y, Xtest=X_test, ytest=y_test, k=k,
                                                      sn2=sn2, mu=mu)

    if args[ALGORITHM] in ("GPyTorchExactGPR", "GPyTorchVariationalGPR"):
        named_variables = {n: v for n, v in algorithm.named_parameters()}
    else:
        named_variables = {n: v for n, v in itertools.chain(k.named_parameters(), algorithm.get_named_tunable_parameters())}
        named_variables['raw_sn2'] = raw_sn2
        named_variables['raw_mu'] = raw_mu

    options = dict(ftol=0., gtol=0., factr=0., disp=args["debug_level"] == DEBUG)
    log(level=INFO, msg=f"running optimizer on {dataset} with {algorithm.get_name()}")
    result = Scipy().minimize(closure, named_variables, algorithm, step_callback=record_metrics_callback,
                            max_time=max_time, method=optimizer, max_iter=max_iterations, scipy_options=options)
    mlflow.set_tag(OPTIMIZATION_MSG, result.message)
    mlflow.end_run()
    # in case someone wants this
    return result, algorithm


def _instantiate_noise(args, device, dtype=torch.float64):
    lower_noise_constraint = args[LOWER_NOISE_CONSTRAINT]
    mlflow.set_tag(LOWER_NOISE_CONSTRAINT, lower_noise_constraint)
    lower_noise_constraint = torch.tensor(lower_noise_constraint, requires_grad=False, device=device, dtype=dtype)
    #raw_sn2 = Parameter(torch.sqrt(torch.tensor(args["initial_noise"], dtype=torch.float64, device=device))) - lower_noise_constraint
    #sn2 = lambda: torch.square(raw_sn2) + lower_noise_constraint
    # to reproduce the results from Artemev et al., we use the same noise transform
    raw_sn2 = inv_softplus(Parameter(torch.tensor(args["initial_noise"], device=device, dtype=dtype)) - lower_noise_constraint)
    sn2 = _make_noise_func(raw_sn2, lower_noise_constraint)
    return raw_sn2, sn2


def _make_noise_func(raw_sn2, lower_noise_constraint):
    return lambda: torch.nn.functional.softplus(raw_sn2) + lower_noise_constraint


def _instantiate_mean(args, device, dtype=torch.float64):
    raw_mu = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
    mu = _make_mean_func(raw_mu)
    return raw_mu, mu


def _make_mean_func(raw_mu):
    return lambda X: raw_mu


def generate_large_scale_batch_jobs(**args):
    #raise NotImplementedError("this setting needs more memory for error prediction!")
    #seeds = [999] #, 888]  #, 777, 0]  # using the same seeds as Artemev et al.
    seeds = list(range(5))

    #ks = KernelFactory().get_available_kernel_functions()
    def get_algorithm_commands(optimizer: str = "BFGS"):
        algo_commands = [
                         f" --{ALGORITHM} {StoppedCholesky.get_registry_key()} -ibs {INITIAL_BLOCK_SIZE} -bs {BLOCK_SIZE} -e all_points --{OPTIMIZER} {optimizer}",
                         ]
        return algo_commands

    device_and_datasets = {
        "cpu":  ["usflight"]
    }

    for device, datasets in device_and_datasets.items():
        k = "ArtemevExp"  # use BFGS and L-BFGS-B
        #commands = _get_commands_for_device(device, get_algorithm_commands("BFGS"), seeds, datasets, k)
        commands = _get_commands_for_device(device, get_algorithm_commands("L-BFGS-B"), seeds, datasets, k)
        _run_commands_on_device(commands, device, scheduler=args["scheduler"])


def generate_batch_jobs(**args):
    seeds = list(range(5))

    def get_algorithm_commands(optimizer: str):
        algo_commands = [f" --{ALGORITHM} {ExactGPR.get_registry_key()} --{OPTIMIZER} {optimizer}",
                         f" --{ALGORITHM} {StoppedCholesky.get_registry_key()} -ibs {INITIAL_BLOCK_SIZE} -bs {BLOCK_SIZE} -e all_points --{OPTIMIZER} {optimizer}",
                         # f" --{ALGORITHM} {CGLB.get_registry_key()} -ni 2048 -cgs 100 --no-{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}"
                         # f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 512 --no-{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                         # f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 1024 --no-{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}"
                         # f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 2048 --no-{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}"
                        ]
        if optimizer != "BFGS":
            # can't optimize over so many parameters with BFGS
            algo_commands += [
                f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 512 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 1024 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 2048 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {NativeVariationalGPR.get_registry_key()} -ni 4096 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}"
                f" --{ALGORITHM} {CGLB.get_registry_key()} -ni 512 -cgs 100 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {CGLB.get_registry_key()} -ni 1024 -cgs 100 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {CGLB.get_registry_key()} -ni 2048 -cgs 100 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}",
                f" --{ALGORITHM} {CGLB.get_registry_key()} -ni 4096 -cgs 100 --{OPTIMIZE_INDUCING_INPUTS}  --{OPTIMIZER} {optimizer}"
            ]
        return algo_commands

    device_and_datasets = {
        "cuda": ["wilson_bike", "wilson_pol", "wilson_elevators", "wilson_pumadyn32nm"],
        "cpu":  ["protein", "wilson_kin40k", "metro", "pm25"] #, "wilson_keggdirected"]  # TODO: if there is more cluster time for more experiments
        #"cpu_small": ["protein", "wilson_kin40k", "metro", "pm25"]
    }

    for device, datasets in device_and_datasets.items():
        k = "ArtemevExp"  # use BFGS and L-BFGS-B
        commands = _get_commands_for_device(device, get_algorithm_commands("L-BFGS-B"), seeds, datasets, k)
        _run_commands_on_device(commands, device, scheduler=args["scheduler"])
        continue
        commands = _get_commands_for_device(device, get_algorithm_commands("BFGS"), seeds, datasets, k)
        _run_commands_on_device(commands, device, scheduler=args["scheduler"])
        k = "InducingPlus"  # use only L-BFGS
        commands = _get_commands_for_device(device, get_algorithm_commands("L-BFGS-B"), seeds, datasets, k)
        _run_commands_on_device(commands, device, scheduler=args["scheduler"])


def _get_commands_for_device(device, algo_commands, seeds, datasets, k):
    template = f"python {__file__} -m execute_single_batch_job --{DEVICE} {device}  --{MAX_ITERATIONS} 2000"
    commands = []
    for seed in seeds:
        for dataset in datasets:
            # make sure the experiment exists to avoid conflicts
            experiment_name = device + HYPER_PARAMETER_TUNING_EXPERIMENT + dataset + k
            mlflow.set_experiment(experiment_name)
            command = template + " -en %s -d %s" % (experiment_name, dataset)
            command += " --" + KERNEL_NAME + " " + k
            command += " -s %i" % seed
            for algo_command in algo_commands:
                commands.append(command + algo_command)
    return commands


def _run_commands_on_device(commands, device, scheduler="slurm"):
    if device == "cpu":
        cpus = 40
        memory = 50000
        exclusive = True
        max_jobs_parallel = 7
        core_affinity = True
    elif device == "cpu_small":
        cpus = 8
        memory = 50000
        exclusive = False
        max_jobs_parallel = 25
        device = "cpu"  # this is what we have to hand over to the scheduler
        core_affinity = False
    elif device == "cuda":
        cpus = 8
        memory = 20000
        exclusive = False
        max_jobs_parallel = 20
        core_affinity = False
    else:
        raise RuntimeError(f"Unknown device: {device}")

    if scheduler == "slurm":
        execute_job_array_on_slurm_cluster(commands, cpus=cpus, mem=memory, device=device, exclusive=exclusive,
                                           max_jobs_parallel=max_jobs_parallel, set_core_affinity=core_affinity)
    elif scheduler == "lsf":
        cpus = 16
        memory = 120000
        max_jobs_parallel = 99
        execute_job_array_on_lsf_cluster(commands, cpus=cpus, mem=memory,
            device=device, exclusive=exclusive, max_jobs_parallel=max_jobs_parallel)
    else:
        raise ValueError(f"Unknown scheduler '{scheduler}'.")


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--mode", type=str, choices=["generate_batch_jobs",
                                                           "execute_single_batch_job",
                                                           "generate_large_scale_batch_jobs"],
                        #default="generate_batch_jobs")
                        default="execute_single_batch_job")
    parser.add_argument("-d", "--" + DATASET, type=str, default="pumadyn")

    parser.add_argument("-mi", "--" + MAX_ITERATIONS, type=int, default=25)
    parser.add_argument("-mt", "--" + MAX_TIME, type=float, default=np.infty)

    parser.add_argument("-v", "--verbose", type=bool, default=False)

    parser.add_argument("-en", "--experiment-name", type=str, default="debug")#None)

    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-ln", "--" + LOWER_NOISE_CONSTRAINT, type=float, default=1e-6)

    parser.add_argument("-in", "--initial-noise", type=float, default=1.0)
    available_kernel_functions = KernelFactory().get_available_kernel_functions()
    parser.add_argument("-kn", "--" + KERNEL_NAME, type=str, choices=available_kernel_functions,
                        default=available_kernel_functions[-3])
    KernelFactory().add_parameters_to_parser(parser)

    parser.add_argument("-a", "--" + ALGORITHM, type=str, choices=HYPER_PARAMETER_TUNING_ALGOS.keys(),
                        #default=StoppedCholesky.get_registry_key())
                        #default=ExactGPR.get_registry_key())
                        #default=CGLB.get_registry_key())
                        default=NativeVariationalGPR.get_registry_key())
                        #default=VariationalGPR.get_registry_key())
    for a in HYPER_PARAMETER_TUNING_ALGOS.keys():
        HYPER_PARAMETER_TUNING_ALGOS[a].add_parameters_to_parser(parser)

    parser.add_argument("-o", "--" + OPTIMIZER, type=str, choices=["BFGS", "L-BFGS-B"], default="L-BFGS-B")

    parser.add_argument("-dv", "--" + DEVICE, type=str, choices=["cpu", "cuda", "cpu_small"], default="cpu")

    parser.add_argument("-dl", "--debug_level", type=int, choices=[DEBUG, INFO, WARNING, ERROR], default=DEBUG)
    parser.add_argument("--scheduler", type=str, choices=["slurm", "lsf"],
        default="slurm")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
