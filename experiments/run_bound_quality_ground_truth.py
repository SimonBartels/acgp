"""
Script to create data for which we can look at our bounds.
"""
from time import thread_time, time

from external.robustgp.inducing_input_init import ConditionalVariance
from utils.registry import KERNEL_DICT, ENVIRONMENT_DICT
from utils.execution.run_cluster import execute_job_array_on_slurm_cluster
from bound_quality.parser import get_parser
from utils.result_management.constants import SETUP_TIME, SETUP_TIME_WC


def main(**args):
    if args["mode"] == "execute_single_batch_job":
        execute_single_batch_job(**args)
    else:
        generate_batch_jobs(**args)


def execute_single_batch_job(**args):
    import numpy as np
    import random
    import gc
    import mlflow
    import numpy.random

    from acgp.blas_wrappers.openblas.openblas_wrapper import OpenBlasWrapper
    from utils.hooks.record_hook import RecordHook
    from utils.kernel_evaluator import get_kernel_evaluator
    from acgp.meta_cholesky import MetaCholesky
    from utils.data.load_dataset import load_dataset
    from utils.result_management.result_management import define_run, set_experiment_tags, get_bound_results_path
    mlflow.set_tracking_uri(get_bound_results_path())

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
    y = np.asfortranarray(y[p, :])
    A = np.zeros([N, N], order='F')
    #A = k.K(X)
    #np.fill_diagonal(A, np.diag(A) + sn2)

    #block_size = 256 * ENVIRONMENT_DICT[ENV_CPUS]
    block_size = args["block_size"]
    chol = MetaCholesky(block_size=block_size, initial_block_size=block_size, blaswrapper=OpenBlasWrapper())
    k = KERNEL_DICT[args["kernel"]].initialize_from_parser(args)

    mlflow.start_run()
    define_run(chol, chol.parameters, seed)
    # we must have defined the run first, to have an active run object...
    set_experiment_tags(dataset, k, ENVIRONMENT_DICT, sn2)

    gc.collect()
    gc.disable()
    M = args["preconditioner_steps"]
    mlflow.set_tag("preconditioner_steps", M)

    t0_wc = time()
    t0 = thread_time()
    if M > 0:
        def kernel_wrapper(x1, x2, full_cov):
            if not full_cov:
                return k.Kdiag(x1)
            else:
                return k.K(x1, x2)
        _, indices = ConditionalVariance(
            X, M, kernel_wrapper
        )
        N = X.shape[0]
        perm = np.arange(N)
        perm[:M] = indices
        perm[M:] = np.setdiff1d(np.arange(N), indices)
        # TODO: the following copy can be avoided, but it shouldn't be too bad
        X = X[perm, :]
        y = y[perm, :]
    mlflow.log_metric(SETUP_TIME, thread_time() - t0)
    mlflow.log_metric(SETUP_TIME_WC, time() - t0_wc)

    chol.run_configuration(A, y, kernel_evaluator=get_kernel_evaluator(X=X, k=lambda *args: k.K(*args), sn2=sn2),
                           hook=RecordHook(N=N, seed=seed, preconditioner_size=M))
    mlflow.end_run()


def generate_batch_jobs(**args):
    from utils.result_management.result_management import initialize_experiment

    cpus = 40

    seeds = range(5)
    datasets = ['metro', 'pm25', 'protein', 'wilson_kin40k']
    ls = [-1., 0., 1., 2.]

    ks = list(KERNEL_DICT.keys())

    sn2 = 1e-3
    theta = 0.

    template = "python %s -m execute_single_batch_job" % __file__
    commands = []
    for seed in seeds:
        for l in ls:
            for k in ks:
                kernel = KERNEL_DICT[k].initialize(log_var=theta, log_ls2=l)
                for dataset in datasets:
                    # make sure the experiment exists to avoid conflicts
                    experiment_name = initialize_experiment(dataset=dataset, kernel=kernel, sn2=sn2, cpus=cpus)
                    command = template + " -en %s -d %s" % (experiment_name, dataset)
                    command += " -k " + kernel.name + kernel.generate_command_string()
                    command += " -sn2 %f" % sn2
                    command += " -s %i" % seed
                    # this is the ground_truth experiment
                    commands.append(command + " -bs 50000")
                    # this is the vanilla stopped cholesky
                    command += f" -bs {cpus * 256}"
                    commands.append(command)
                    #command += f" -ps 2048"
                    # stopped cholesky with preconditioning
                    #commands.append(command)
                    #cluster_command = execute_single_configuration_on_slurm_cluster(command=command, cpus=cpus)
                    #print("executing: %s" % cluster_command)
                    #print("with: %s" % command)
                    #run_local(command)
    print(execute_job_array_on_slurm_cluster(commands, cpus=cpus))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(**vars(args))
