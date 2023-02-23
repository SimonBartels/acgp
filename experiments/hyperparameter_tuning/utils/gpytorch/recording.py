import math
from warnings import warn

import gpytorch.kernels
import mlflow
import torch

from utils.result_management.constants import PREDICTION_TIME, RMSE, NLPD, PARAMETERS, PREDICTION_TIME_WC
from utils.result_management.result_management import get_results_path
from hyperparameter_tuning.utils.abstract_hyper_parameter_tuning_algorithm import AbstractHyperParameterTuningAlgorithm

from time import time
try:
    from time import thread_time as thread_time
except:
    warn("Failed to import thread_time! Going to use default time which might give different results.")
    thread_time = time

assert(mlflow.get_tracking_uri() == get_results_path())


def make_recording_callback(algorithm: AbstractHyperParameterTuningAlgorithm, X: torch.Tensor, y: torch.Tensor,
                            Xtest: torch.Tensor, ytest: torch.Tensor, k: gpytorch.kernels.Kernel, sn2: callable,
                            mu: callable):
    c = torch.log(2 * torch.tensor(math.pi)) / 2
    def record_metrics_callback(step: int, variables: dict):
        mlflow.log_dict({n: v.cpu().detach().tolist() for n, v in variables.items()}, PARAMETERS + str(step))
        # for n, v_ in variables.items():
        #     t = v_.flatten().detach().tolist()
        #     for i, v in enumerate(t):
        #         mlflow.log_metric(n + '_' + str(i), v, step=step)

        algorithm.k.train(False)
        t0 = time()
        tt0 = thread_time()
        mu, var = algorithm.get_posterior(Xtest)
        mu, var = algorithm.get_y_posterior(Xtest, mu, var)
        pred_time = thread_time() - tt0
        pred_time_wc = time() - t0
        algorithm.k.train(True)
        mlflow.log_metric(PREDICTION_TIME, pred_time, step=step)
        mlflow.log_metric(PREDICTION_TIME_WC, pred_time_wc, step=step)
        assert(mu.shape[1] == 1)
        assert(var.shape[1] == 1)
        rmse = torch.sqrt(torch.mean(torch.square(mu - ytest)))
        nlpd = torch.mean(torch.square(mu - ytest) / var + torch.log(var)) / 2 + c
        mlflow.log_metric(RMSE, rmse.item(), step=step)
        mlflow.log_metric(NLPD, nlpd.item(), step=step)
    return record_metrics_callback
