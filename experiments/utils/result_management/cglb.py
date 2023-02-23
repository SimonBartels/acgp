import numpy as np

from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import NUM_INDUCING_INPUTS
from utils.result_management.constants import ALGORITHM, U_DET, L_DET, U_QUAD, L_QUAD, STEP_TIME, SETUP_TIME


def process_cglb_runs(runs):
    result_dict = {}
    for r in runs:
        if r.data.tags[ALGORITHM].startswith("cglb"):
            (
                idx,
                inducing_points,
                times,
                log_det_upper_bounds,
                log_det_lower_bounds,
                quad_upper_bounds,
                quad_lower_bounds,
            ) = process_cglb_run(r)

            result_dict[(idx, times)] = (
                inducing_points,
                log_det_upper_bounds,
                log_det_lower_bounds,
                quad_upper_bounds,
                quad_lower_bounds,
            )
    bounds = result_dict.values()
    inducing_points = [x for x, _, _, _, _ in bounds]
    log_det_upper_bounds = [x for _, x, _, _, _ in bounds]
    log_det_lower_bounds = [x for _, _, x, _, _ in bounds]
    quad_upper_bounds = [x for _, _, _, x, _ in bounds]
    quad_lower_bounds = [x for _, _, _, _, x in bounds]
    idx = [x for x, _ in result_dict.keys()]
    times = [x for _, x in result_dict.keys()]
    return (
        idx,
        inducing_points,
        times,
        log_det_upper_bounds,
        log_det_lower_bounds,
        quad_upper_bounds,
        quad_lower_bounds,
    )


# load CGLB data
def process_cglb_run(run):
    # run_id = run.info.run_id
    log_det_upper_bounds = run.data.metrics[U_DET]
    log_det_lower_bounds = run.data.metrics[L_DET]
    quad_upper_bounds = run.data.metrics[U_QUAD]
    quad_lower_bounds = run.data.metrics[L_QUAD]
    times = run.data.metrics[STEP_TIME] + run.data.metrics[SETUP_TIME]
    inducing_points = int(run.data.tags[ALGORITHM + "." + NUM_INDUCING_INPUTS])
    print(inducing_points)
    print(log_det_upper_bounds)
    idx = None  #np.power(np.square(inducing_points) * N, 1.0 / 3.0)
    return (
        idx,
        inducing_points,
        times,
        log_det_upper_bounds,
        log_det_lower_bounds,
        quad_upper_bounds,
        quad_lower_bounds,
    )
