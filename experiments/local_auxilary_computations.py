import warnings
import mlflow
import numpy as np
from mlflow.entities import Experiment
from warnings import warn
import torch
import logging

from utils.data.load_dataset import get_train_test_dataset
from hyperparameter_tuning.utils.gpytorch.models.exact_gpr import ExactGPR
from utils.result_management.constants import EXPERIMENT_TYPE, HYPER_PARAMETER_TUNING_EXPERIMENT, SEED, PARSER_ARGS, \
    ALGORITHM, DATASET, KL_CURR_FINAL, KL_FINAL_CURR, EXACT_LOG_DET, EXACT_QUAD, PARAMETERS, APPROXIMATE_LOSS, RMSE, \
    EXACT_LOSS
from utils.result_management.result_management import get_results_path, get_steps_and_values_from_run, load_artifact_dict

assert(mlflow.get_tracking_uri() == get_results_path())


def _attempt_data_repair(steps, values):
    multiples = _get_multiples(steps)
    popped_elements = 0
    for s, ls in multiples:
        id0 = ls[0] - popped_elements
        val = values[id0]
        step = steps[id0]
        for i in range(1, len(ls)):
            id = ls[i] - popped_elements
            #assert(values[id] == val)
            assert(steps.pop(id) == step)
            assert(values.pop(id) == val)
            popped_elements += 1


def _get_multiples(steps):
    back_map = {x: [] for x in range(max(steps)+1)}
    [back_map[x].append(i) for (i, x) in enumerate(steps)]
    return [(x, l) for x, l in back_map.items() if len(l) > 1]


def run_local_auxiliary_computations(experiment_id: str) -> ():
    """
    Fill EXACT_LOSS field from the computed values.
    :param experiment_id:
    :return:
    """
    mlfc = mlflow.tracking.MlflowClient()
    exp: Experiment = mlfc.get_experiment(experiment_id)
    if not exp.tags[EXPERIMENT_TYPE] == HYPER_PARAMETER_TUNING_EXPERIMENT:
        warn("Experiment is not a hyper-parameter tuning experiment. Abort.")
        return
    dataset = exp.tags[DATASET]
    X, _, _, _ = get_train_test_dataset(dataset)
    c = X.shape[0] / 2 * np.log(2 * np.pi)
    del X
    all_runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")

    with torch.no_grad():
        for i, r in all_runs.iterrows():
            run_id = r["run_id"]
            algorithm = r["tags." + ALGORITHM]
            if algorithm == ExactGPR.get_registry_key():
#                try:
                    steps, _ = get_steps_and_values_from_run(run_id, RMSE)
                    _, losses = get_steps_and_values_from_run(run_id, APPROXIMATE_LOSS)
                    for i in range(len(steps)):
                        mlfc.log_metric(run_id, EXACT_LOSS, losses[steps[i]-1], step=steps[i])
#                except Exception as e:
#                    logging.error(e)
            else:
                try:
                    steps_, quads = get_steps_and_values_from_run(run_id, EXACT_QUAD)
                    steps, dets = get_steps_and_values_from_run(run_id, EXACT_LOG_DET)
                    if len(steps) != len(steps_):
                        # remove duplicates
                        _attempt_data_repair(steps, dets)
                        _attempt_data_repair(steps_, quads)
                        # take intersection
                        if len(steps) != len(steps_):
                            assert(abs(len(steps) - len(steps_)) == 1)  # probably the job was killed in the middle
                            min_length = min(len(steps), len(steps_))
                            steps = steps[:min_length]
                            steps_ = steps_[:min_length]
                            dets = dets[:min_length]
                            quads = quads[:min_length]
                        assert(steps == steps_)
                        assert(len(steps) == len(dets) == len(quads))
                    for i in range(len(steps)):
                        v = quads[i] / 2 + dets[i] / 2 + c
                        mlfc.log_metric(run_id, EXACT_LOSS, v, step=steps[i])
                except Exception as e:
                    warnings.warn(f"exception for {algorithm} on {dataset} (seed {r['tags.' + SEED]})")
                    logging.error(e)


if __name__ == "__main__":
    # TODO: start at 1 again
    experiment_ids = [str(i) for i in range(1, 9)]
    for e in experiment_ids:
        run_local_auxiliary_computations(e)
        logging.info("done processing experiment %s" % e)
    #exit()
