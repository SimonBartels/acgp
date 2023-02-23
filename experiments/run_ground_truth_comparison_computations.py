"""
This script computes post-optimization ground-truth quantities for which the approximation algorithm itself is not needed.
This saves memory and it allows to run experiments independently.
"""
import mlflow
import numpy as np
from mlflow.entities import Experiment
from warnings import warn
import torch
import logging

from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import StoppedCholesky
from run_hyper_parameter_tuning import _make_noise_func, _make_mean_func, LOWER_NOISE_CONSTRAINT, OPTIMIZER
from utils.data.load_dataset import get_train_test_dataset
from hyperparameter_tuning.utils.gpytorch.models.exact_gpr import ExactGPR
from hyperparameter_tuning.utils.gpytorch.kernel_factory import KernelFactory
from utils.execution.run_cluster import execute_job_array_on_slurm_cluster
from utils.result_management.constants import EXPERIMENT_TYPE, HYPER_PARAMETER_TUNING_EXPERIMENT, SEED, PARSER_ARGS, \
    ALGORITHM, DATASET, KL_CURR_FINAL, KL_FINAL_CURR, EXACT_LOG_DET, EXACT_QUAD, PARAMETERS, APPROXIMATE_LOSS, RMSE, \
    EXACT_LOSS
from utils.result_management.result_management import get_results_path, get_steps_and_values_from_run, load_artifact_dict

assert(mlflow.get_tracking_uri() == get_results_path())


def add_ground_truth_values_to_experiment(experiment_id: str) -> ():
    """
    This is the main function. For a given experiment id this function
     * collects the ground-truth runs
     * computes a couple of diagnostics
     * then iterates over all ALL runs (including ground-truth)
     * and logs metrics as if they were computed during the optimization process of each run.
    :param experiment_id:
        the experiment id for which to post-compute metrics
    :return:
        None
    """
    mlfc = mlflow.tracking.MlflowClient()
    exp: Experiment = mlfc.get_experiment(experiment_id)
    if not exp.tags[EXPERIMENT_TYPE] == HYPER_PARAMETER_TUNING_EXPERIMENT:
        warn("Experiment is not a hyper-parameter tuning experiment. Abort.")
        return

    all_runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")
    seeds = set(all_runs["tags." + SEED])
    if None in seeds:
        seeds.remove(None)  # ignore crashed runs
    seeds = sorted(seeds)
    dataset = exp.tags[DATASET]
    ground_truth_runs = all_runs[(all_runs["tags." + ALGORITHM] == ExactGPR.get_registry_key())]

    command_ls = []
    command_template = "python %s " % __file__
    with torch.no_grad():
        for seed in seeds:
            if int(seed) > 5:
                warn("Ignoring seeds larger than 5!")
                continue
            gtr = ground_truth_runs[(ground_truth_runs["tags." + SEED] == seed)]
            if gtr.shape[0] == 0:
                continue  # seems like ground truth results for this seed are still running
            elif gtr.shape[0] > 1:
                warn(f"Found more than one ground truth result for seed {seed}") #--skipping this seed.")
            gtr = list(gtr.iterrows())[0][1]

            runs = all_runs[(all_runs["tags." + SEED] == seed)]
            # now I need the metric history for each parameter...
            # TODO: Argh, maybe this can not really be parallelized--at least not on one computer
            #pool = multiprocessing.Pool(4)
            #pool.map(_compute_metrics_against_ground_truth_for_run, runs.iterrows())
            for i, r in runs.iterrows():
                if r["tags." + ALGORITHM] == ExactGPR.get_registry_key():
                    continue
                if r["tags." + OPTIMIZER] != "L-BFGS-B":
                    # currently we need the results of those runs first
                    continue
                # if r["tags." + ALGORITHM] != StoppedCholesky.get_registry_key() or r["tags.mlflow.source.git.commit"] != "b618190cc7227e640b39acb37055770dacd67ce2":
                #     continue
                try:
                    #_compute_metrics_against_ground_truth_for_run(experiment_id, r["run_id"], gtr["run_id"], dataset, seed)
                    command_ls.append(command_template + experiment_id + " " + r["run_id"] + " " + gtr["run_id"] + " " + dataset + " " + seed)
                except Exception as e:
                    logging.error(e)
    execute_job_array_on_slurm_cluster(command_ls, cpus=8, exclusive=False, max_jobs_parallel=4, set_core_affinity=False)


def _get_metric_history(run_id: str, metric: str):
    _, values = get_steps_and_values_from_run(run_id=run_id, metric=metric)
    return values


@torch.no_grad()
def _compute_metrics_against_ground_truth_for_run(experiment_id, run_id, ground_truth_run_id, dataset, seed):
    steps, losses = get_steps_and_values_from_run(run_id, RMSE)  # get steps for accepted steps
    try:
        already_computed_steps, true_losses = get_steps_and_values_from_run(run_id, EXACT_QUAD)
    except:
        already_computed_steps = []

    steps = np.setdiff1d(steps, already_computed_steps)

    # TODO: remove
    # compute ground truth for last step
    #if len(steps) > 1:
    #    steps = [steps[-1]]

    X, y, X_test, y_test = get_train_test_dataset(dataset, seed=int(seed))
    X = torch.tensor(X)
    y = torch.tensor(y)
    X_test = torch.tensor(X_test)
    y_test = torch.tensor(y_test)
    general_variables = {"X": X, "y": y, "X_test": X_test, "y_test": y_test}

    original_parser_args = load_artifact_dict(experiment_id=experiment_id, run_id=ground_truth_run_id,
                                              artifact_name=PARSER_ARGS)
    k = KernelFactory().create(args=original_parser_args, X=X)

    with torch.no_grad():
        for step in steps:
            params = load_artifact_dict(experiment_id=experiment_id, run_id=run_id, artifact_name=PARAMETERS + str(step))
            raw_sn2 = torch.tensor(params.pop("raw_sn2"))
            lower_noise_constraint = torch.tensor(float(original_parser_args[LOWER_NOISE_CONSTRAINT]), dtype=torch.float64)
            sn2 = _make_noise_func(raw_sn2, lower_noise_constraint=lower_noise_constraint)
            raw_mu = torch.tensor(params.pop("raw_mu"))
            mu = _make_mean_func(raw_mu)

            for n, _ in k.named_parameters():
                p = torch.nn.Parameter(torch.tensor(params[n], dtype=torch.float64, requires_grad=False))
                # setattr doesn't work due to dots in the name
                exec(f"k.{n} = p")
            _register_metrics_in_step(step, run_id, k, sn2, mu, **general_variables)


@torch.no_grad()
def _register_metrics_in_step(step, run_id, k, sn2, mu, X, y, X_test, y_test):
    mlfc = mlflow.tracking.MlflowClient()
    N = X.shape[0]
    L = k(X).evaluate() + sn2() * torch.eye(N)
    torch.cholesky(L, out=L)
    #mu = torch.triangular_solve(torch.triangular_solve(y, L, upper=False).solution, L.T, upper=True)
    det = 2 * torch.sum(torch.log(torch.diag(L)))
    mlfc.log_metric(run_id, EXACT_LOG_DET, det.item(), step=step)
    alpha = torch.triangular_solve(y-mu(X), L, upper=False).solution
    quad = torch.sum(torch.square(alpha))
    mlfc.log_metric(run_id, EXACT_QUAD, quad.item(), step=step)


def generate_batch_jobs():
    command_ls = []
    exp_ls = mlflow.list_experiments()
    template = "python %s " % __file__
    #warn("Considering only large-scale CPU experiments!")
    for e in exp_ls:
        try:
            if e.tags[EXPERIMENT_TYPE] == HYPER_PARAMETER_TUNING_EXPERIMENT:
                #if (e.tags[DATASET] in ["wilson_kin40k", "protein", "metro", "pm25"] and e.name.startswith("cpuh")) or e.name.startswith("cuda"):
                    command_ls.append(template + e.experiment_id)
        except Exception as e:
            logging.error(e)
            continue
    # no need to run exclusive experiments
    if len(command_ls) > 0:
        execute_job_array_on_slurm_cluster(command_ls, cpus=8, exclusive=False, max_jobs_parallel=8, set_core_affinity=False)
    else:
        warn("Nothing to run")


if __name__ == "__main__":
    from sys import argv
    if len(argv) == 2:
        add_ground_truth_values_to_experiment(argv[1])
    elif len(argv) > 2:
        _compute_metrics_against_ground_truth_for_run(*argv[1:])
    else:
        generate_batch_jobs()
