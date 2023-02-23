from logging import log, DEBUG

import torch
import scipy.optimize
from warnings import warn
import numpy as np
import mlflow

from utils.result_management.constants import LOSS_TIME, APPROXIMATE_LOSS, GRAD_TIME, LOSS_TIME_WC, GRAD_TIME_WC, \
    LOSS_TIME_PS, GRAD_TIME_PS
from utils.result_management.result_management import get_results_path
from hyperparameter_tuning.utils.gpytorch.scipy import Scipy as CGLBScipy

from time import time
try:
    from time import thread_time as thread_time
    from time import process_time as process_time
except:
    warn("Failed to import thread_time! Going to use default time which might give different results.")
    thread_time = time

assert(mlflow.get_tracking_uri() == get_results_path())


class Scipy(CGLBScipy):
    # TODO: Simon: This function deviates from the super-class signature! Refactor!
    def minimize(self, closure, named_variables, algorithm, method="L-BFGS-B", step_callback=None, max_time=np.infty,
                 max_iter=2000, scipy_options={}):
        variables = tuple(named_variables.values())
        last_accepted_step = self.to_numpy(self.pack(variables)).copy()


        step = 0
        cum_time = 0

        def _compute_loss_and_gradients(loss_closure, variables):
            nonlocal step
            nonlocal cum_time
            step += 1
            t0 = time()
            pt0 = process_time()
            tt0 = thread_time()
            loss = loss_closure()
            loss_time = thread_time() - tt0
            loss_time_ps = process_time() - pt0
            loss_time_wc = time() - t0
            mlflow.log_metric(LOSS_TIME, loss_time, step=step)
            mlflow.log_metric(LOSS_TIME_PS, loss_time_ps, step=step)
            mlflow.log_metric(LOSS_TIME_WC, loss_time_wc, step=step)
            mlflow.log_metric(APPROXIMATE_LOSS, loss.item(), step=step)
            log(level=DEBUG, msg=f"loss: {loss.item()}")

            t0 = time()
            pt0 = process_time()
            tt0 = thread_time()
            grads = torch.autograd.grad(loss, variables)
            grad_time = thread_time() - tt0
            grad_time_ps = process_time() - pt0
            grad_time_wc = time() - t0
            mlflow.log_metric(GRAD_TIME, grad_time, step=step)
            mlflow.log_metric(GRAD_TIME_PS, grad_time_ps, step=step)
            mlflow.log_metric(GRAD_TIME_WC, grad_time_wc, step=step)

            algorithm.log_metrics(step)

            cum_time += loss_time + grad_time

            if cum_time > max_time:
                # this way we force the optimization to stop
                loss = torch.finfo.min
                grads = torch.zeros_like(grads)
            return loss, grads

        def eval_func(closure, variables):
            device = variables[0].device

            def _torch_eval(x):
                values = self.unpack(variables, x)
                self.assign(variables, values)

                loss, grads = _compute_loss_and_gradients(closure, variables)
                return loss, self.pack(grads)

            def _eval(x):
                loss, grad = _torch_eval(torch.from_numpy(x).to(device))
                return (
                    loss.cpu().detach().numpy().astype(np.float64),
                    grad.cpu().detach().numpy().astype(np.float64),
                )

            return _eval

        def callback_func(step_callback):
            def _callback(x):
                last_accepted_step[:] = x[:]
                nonlocal step
                step_callback(step, named_variables)
            return _callback

        callback = None
        if step_callback is not None:
            callback = callback_func(step_callback)
            #scipy_kwargs.update(dict(callback=callback))

        # TODO: it appears Artemev restarts optimization (up to 3 times) because BFGS stops!
        # TODO: Also, it seems that at some point inducing inputs are not optimized
        result = None
        attempt = 0
        while max_iter > 0:
            # TODO: it would be better if the option below was maxfun!
            # TODO: though, then we have to mention this as a deviation from the Artemev et al. experiments
            scipy_options["maxiter"] = max_iter
            scipy_options["ftol"] = algorithm.optimization_strategy().get_tolerance(result, attempt)
            init_vals = last_accepted_step.copy()
            result = scipy.optimize.minimize(eval_func(closure, variables), init_vals, jac=True, method=method,
                                             callback=callback, options=scipy_options)
            attempt += 1
            if algorithm.optimization_strategy().abort(result, attempt):
                break
            max_iter = max_iter - result.nit

        return result
