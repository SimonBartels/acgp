import numpy as np
import pickle
from builtins import FileNotFoundError
import warnings

# this import is necessary to set the tracking uri correct
from utils.result_management.result_management import get_steps_and_values_from_run
from acgp.hooks.stop_hook import StopHook
from utils.result_management.constants import EXACT_SOLUTIONS, DIAGONAL, EQSOL, TEMP_VALUES, TEMP, OFFDIAGONAL, ALGORITHM, STEP_TIME, SETUP_TIME


def fix_file_name(file_name):
    return file_name.split("experiments/")[1]


def process_stopped_chol_run(run, sn2):
    # fill the hook
    file_name = fix_file_name(run.data.tags[EXACT_SOLUTIONS])
    d = pickle.load(open(file_name, "rb"))
    diagL = np.squeeze(d[DIAGONAL])
    assert len(diagL.shape) == 1
    alpha = np.squeeze(d[EQSOL])
    assert diagL.shape == alpha.shape
    log_det = 2 * np.sum(np.log(diagL))
    quad = np.sum(np.square(alpha))

    try:
        file_name = fix_file_name(run.data.tags[TEMP_VALUES])
        d = pickle.load(open(file_name, "rb"))
    except FileNotFoundError:
        alternative_file_name = run.data.tags[TEMP_VALUES].replace("results", "results_bounds")
        d = pickle.load(open(alternative_file_name, "rb"))
        warnings.warn("Found a somewhat older result referencing to data in the wrong place.")

    temp_diagK = np.squeeze(d[TEMP + DIAGONAL])
    temp_offdiagK = np.squeeze(d[TEMP + OFFDIAGONAL])
    temp_alpha = np.squeeze(d[TEMP + EQSOL])
    N = diagL.shape[0]
    block_size = int(run.data.tags[ALGORITHM + "." + "block_size"])
    hook = StopHook(N=N, min_noise=sn2)
    # hook.prepare()  # not necessary
    other_auxilary_variables = {
        "average_model_calibration": [],
        "expected_worst_case_increase_rate": [],
        "average_error": [],
        "average_error_overestimate": [],
    }

    K_ = np.zeros([block_size, block_size])
    processed_data = []
    for idi in range(0, N, block_size):
        advance = min(block_size, N - idi)

        processed_data.append(advance + idi)

        if advance < block_size:
            K_ = np.zeros([advance, advance])
        K_ += np.diag(temp_diagK[idi : idi + advance])  # fill diagonal
        K_ += np.diag(temp_offdiagK[idi : idi + advance - 1], -1)  # fill off-diagonal
        y_ = temp_alpha[idi : idi + advance].reshape(-1, 1)  # create solution vector
        hook.pre_chol(idi, K_, y_)
        other_auxilary_variables["average_error_overestimate"].append(
            np.sum(np.square(y_)) / advance
        )
        other_auxilary_variables["average_error"].append(
            np.sum(np.square(diagL[idi : idi + advance] * alpha[idi : idi + advance]))
            / advance
        )

        # now pretend to do the down-date
        K_ *= 0.0
        K_ += np.diag(diagL[idi : idi + advance])  # fill diagonal
        y_ = alpha[idi : idi + advance].reshape(-1, 1)  # create solution vector
        hook.post_chol(idi, K_, y_)
        K_ *= 0.0  # clear matrix

        # other_auxilary_variables["average_model_calibration"].append(np.sum(np.square(alpha[idi:idi+advance])) / advance)
        other_auxilary_variables["average_model_calibration"].append(
            np.sum(
                np.square(diagL[idi : idi + advance] * alpha[idi : idi + advance])
                / temp_diagK[idi : idi + advance]
            )
            / advance
        )
        other_auxilary_variables["expected_worst_case_increase_rate"].append(
            np.sum(
                np.square(
                    diagL[idi : idi + advance - 1]
                    * alpha[idi : idi + advance - 1]
                    * temp_offdiagK[idi : idi + advance - 1]
                    / sn2
                )
            )
            / (advance - 1)
        )

    hook.ldet = log_det
    hook.quad = quad
    hook.finalize()
    nllh, bounds = hook.get_bounds()

    # the first bound estimates are computed before even computing the first Cholesky --
    # it's so bad it screws up the plot
    start_at = 1
    bounds = bounds[start_at:]
    log_det_upper_bounds = [x for x, _, _, _ in bounds]
    log_det_lower_bounds = [x for _, x, _, _ in bounds]
    quad_upper_bounds = [x for _, _, x, _ in bounds]
    quad_lower_bounds = [x for _, _, _, x in bounds]
    # no setup time---that's just allocating N^2 memory which is at most a second
    _, times = get_steps_and_values_from_run(run.info.run_id, STEP_TIME)
    times = np.cumsum(times)[start_at:] + run.data.metrics[SETUP_TIME]
    m = block_size
    # print(-nllh)
    # print(-log_det / 2 - quad / 2 - N * np.log(2 * np.pi) / 2)
    steps = len(log_det_upper_bounds)
    idx = np.arange(start_at, steps + 1) * m
    idx[-1] = N

    return (
        idx,
        processed_data,
        times,
        log_det_upper_bounds,
        log_det_lower_bounds,
        quad_upper_bounds,
        quad_lower_bounds,
        diagL,
        alpha,
        temp_diagK,
        temp_offdiagK,
        temp_alpha
    )
