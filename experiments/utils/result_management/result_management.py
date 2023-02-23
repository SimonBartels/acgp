import yaml
import os
import warnings
from os.path import sep, join
import mlflow
from warnings import warn
import pickle
from mlflow.entities import ViewType
import shutil
from pathlib import Path


from utils.registry import ENVIRONMENT_DICT
from utils.result_management.constants import DATASET, KERNEL, SN2, ENVIRONMENT, ALGORITHM, SEED, ENV_CPUS, ENV_PROC, \
    BLOCK_SIZE, LOG_DET_ESTIMATE, EXACT_SOLUTIONS, TEMP_VALUES, TEMP, DIAGONAL, OFFDIAGONAL, EQSOL, NODE_NAME, RMSE, \
    APPROXIMATE_LOSS


#my_location = os.path.dirname(__file__)
#base_folder = "".join([d + sep for d in my_location.split(sep)[:-3]])  # go up three folders
from hyperparameter_tuning.utils.gpytorch.models.preconditioned_stopped_cholesky import PRECONDITIONER_STEPS

base_folder = os.getcwd()


def _get_base_results_path():
    return os.path.join(base_folder, "output", "results")


def get_results_path():
    return os.path.join(_get_base_results_path(), "mlruns")


def get_bound_results_path():
    #return os.path.join(base_folder, "output", "results_bounds", "mlruns")
    return get_results_path()


# This is the default. Bound experiments will set their result path explicitly.
mlflow.set_tracking_uri(get_results_path())


def apply_binary_operator_to_experiment(f, x, dataset, kernel, environment, sn2):
    x = f(x, DATASET, dataset)
    x = f(x, KERNEL, kernel.name)
    k_params = kernel.get_parameter_dictionary()
    for key in k_params.keys():
        x = f(x, KERNEL + '.' + key, k_params[key])
    for key in environment:
        x = f(x, ENVIRONMENT + '.' + key, environment[key])
    x = f(x, SN2, sn2)
    return x


def apply_binary_operator_to_run(f, x, dataset, kernel, algorithm, algorithm_parameters, environment, sn2, seed):
    x = apply_binary_operator_to_experiment(f, x, dataset, kernel, environment, sn2)
    x = apply_binary_operator_to_run_only(f, x, algorithm, algorithm_parameters, seed)
    return x


def apply_binary_operator_to_run_only(f, x, algorithm, algorithm_parameters, seed):
    x = f(x, ALGORITHM, algorithm.get_signature())
    for key in algorithm_parameters:
        x = f(x, ALGORITHM + '.' + key, algorithm_parameters[key])
    x = f(x, SEED, seed)
    return x


def build_filter_string(*args):
    def f(s, a, b):
        return s + "tag." + a + '="' + str(b) + '" and '
    return apply_binary_operator_to_run(f, "", *args)


def make_experiment_name(*args):
    def f(s, a, b):
        return s+"__"+str(a)+"_"+str(b)
    return apply_binary_operator_to_experiment(f, "", *args)


def get_environment_dict_from_experiment(e: mlflow.entities.experiment):
    return {ENV_CPUS: e.tags[ENVIRONMENT + '.' + ENV_CPUS], ENV_PROC: e.tags[ENVIRONMENT + '.' + ENV_PROC]}


def _has_tag(e, last_check, tag, attribute):
    if not last_check:
        # if the last check failed, we do not need to bother checking further
        return False
    try:
        return e.tags[tag] == str(attribute)
    except KeyError:
        warn("Tag " + tag + " caused an error for experiment with id " + e.experiment_id)
        return False


def find_experiments_with_tags(dataset, kernel, environment, sn2, clip):
    exps = mlflow.tracking.MlflowClient().list_experiments(view_type=ViewType.ALL)
    exps = [e for e in exps if apply_binary_operator_to_experiment(
        lambda *args: _has_tag(e, *args), True, dataset, kernel, environment, sn2, clip)]
    return exps


def find_runs_with_tags(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip, exps=None):
    if exps is None:
        #exp = mlflow.get_experiment_by_name(make_experiment_name(dataset, kernel, environment, sn2, clip))
        exps = find_experiments_with_tags(dataset, kernel, environment, sn2, clip)

    def _build_filter_string(*args):
        def f(s, a, b):
            return s + "tag." + a + '="' + str(b) + '" and '
        return apply_binary_operator_to_run_only(f, "", *args)

    runs = mlflow.tracking.MlflowClient().search_runs(
        experiment_ids=[exp.experiment_id for exp in exps],
        filter_string=_build_filter_string(algorithm, algo_parameters, seed),
        run_view_type=ViewType.ALL)
    return runs


def filter_runs_with_tags(runs, algorithm, algo_parameters, seed):
    def _build_filter_string(*args):
        def f(s, a, b):
            return s & (runs["tags." + a] == b)
        return apply_binary_operator_to_run_only(f, True, *args)
    return runs.loc[_build_filter_string(algorithm, algo_parameters, seed)]


def get_steps_and_values_from_run(run_id: str, metric: str):
    results = mlflow.tracking.MlflowClient().get_metric_history(run_id, metric)
    return [r.step for r in results], [r.value for r in results]


def get_steps_from_config(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip):
    runs = find_runs_with_tags(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip)
    assert(len(runs) <= 1)
    if len(runs) == 0:
        return []
    if runs.shape[1] == 0:
        return []
    steps, _ = get_steps_and_values_from_run(runs[0].info.run_id, LOG_DET_ESTIMATE)
    return steps


def save_large_gpr_quantities(seed, block_size, preconditioner_size, diagL, Linvy, temp_diaglK, temp_offdiagK, temp_y):
    e = mlflow.tracking.MlflowClient().get_experiment(mlflow.active_run().info.experiment_id)
    tags = e.tags
    path_prefix = join(_get_base_results_path(), "diagonals", tags[DATASET] + sep + ''.join([k + "_" + str(tags[k]) + sep for k in sorted(tags.keys()) if k.startswith(KERNEL)]))
    path = join(path_prefix, BLOCK_SIZE + '_' + str(block_size) + '__' + PRECONDITIONER_STEPS + '_' + str(preconditioner_size))
    Path(path).mkdir(parents=True, exist_ok=True)

    file = path + sep + SEED + str(seed) + ".pkl"
    with open(file, "wb+") as f:
        pickle.dump({TEMP + DIAGONAL: temp_diaglK, TEMP + OFFDIAGONAL: temp_offdiagK, TEMP + EQSOL: temp_y}, f)
        f.flush()
        mlflow.set_tag(TEMP_VALUES, file)

    # TODO: Potentially, this write can fail when another, concurrent run does the same.
    # But since this is the very last step of the experiment it should be fine. The results of both runs are the same anyway.
    file = join(path_prefix, PRECONDITIONER_STEPS + '_' + str(preconditioner_size) + '__' + SEED + str(seed) + ".pkl")
    with open(file, "wb+") as f:
        pickle.dump({DIAGONAL: diagL, EQSOL: Linvy}, f)
        f.flush()
        mlflow.set_tag(EXACT_SOLUTIONS, file)


def delete_runs_if_crashed(run_ls: []) -> []:
    deleted_runs = []
    for run in run_ls:
        if not LOG_DET_ESTIMATE in run.data.metrics:
            delete_run(run)
            deleted_runs.append(run)
    remaining = set(run_ls).difference(set(deleted_runs))
    return list(remaining)


def delete_run(run):
    remove = mlflow.get_tracking_uri() + sep + run.info.experiment_id + sep + run.info.run_id
    shutil.rmtree(remove)


def get_run_list_from_dataframe(runs):
    run_ls = []
    for run_id in runs["run_id"].values:
        try:
            run_ls.append(mlflow.tracking.MlflowClient().get_run(run_id))
        except:
            warnings.warn("Could not find run with id %s" % run_id)
    return run_ls


def initialize_experiment(dataset, kernel, sn2, cpus: int=None):
    #assert(mlflow.get_tracking_uri() == get_results_path())
    if cpus is None:
        env_dict = {ENV_CPUS: ENVIRONMENT_DICT[ENV_CPUS]}
    else:
        env_dict = {ENV_CPUS: cpus}
    experiment_name = make_experiment_name(dataset, kernel, env_dict, sn2)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def define_run(*args):
    """
    Adds all necessary tags to the mlflow run.
    """
    def f(s, a, b):
        mlflow.set_tag(a, b)
    apply_binary_operator_to_run_only(f, [], *args)
    mlflow.set_tag(NODE_NAME, os.uname()[1])


def set_experiment_tags(*args):
    """
    Sets the tags for the experiment.
    """
    set_experiment_tags_for(mlflow.active_run().info.experiment_id, *args)


def set_experiment_tags_for(experiment_id, *args):
    """
    Sets tags for a given experiment.
    """
    def f(x, a, b):
        mlflow.tracking.MlflowClient().set_experiment_tag(experiment_id, a, b)
    apply_binary_operator_to_experiment(f, [], *args)


def load_artifact_dict(experiment_id: str, run_id: str, artifact_name: str):
    """
    Loads a dictionary stored with #mlflow.log_dict.
    It's a shame that mlflow does not provide a load function...
    :param experiment_id:
    :param run_id:
    :param artifact_name:
    :return:
    """
    artifact_location = mlflow.get_tracking_uri() + sep + experiment_id + sep + run_id + sep + "artifacts" + sep
    f = open(artifact_location + artifact_name)
    d = yaml.safe_load(f.read())
    f.close()
    return d


def get_last_logged_timestamp(run: mlflow.entities.Run):
    if RMSE in run.data.metrics.keys():
        hist = mlflow.tracking.MlflowClient().get_metric_history(run.info.run_id, RMSE)
        # we assume here that the last logged entry is also the last in time
        # That is not necessarily true for mlflow but in our case this should be the case
        return hist[-1].timestamp
    return run.info.start_time
