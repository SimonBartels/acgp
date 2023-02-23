import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import mlflow

# this import is necessary to set the tracking uri correct
import utils.result_management.result_management as result_management
from hyperparameter_tuning.utils.gpytorch.models.cglb import CGLB
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import NativeVariationalGPR, NUM_INDUCING_INPUTS
from utils.result_management.constants import LOSS_TIME, GRAD_TIME, ALGORITHM, RMSE, DATASET, \
    EXACT_LOSS, SETUP_TIME, NLPD
from utils.result_management.result_management import get_steps_and_values_from_run as get_steps_and_values_from_run_
from make_results_table import checkRunIsIncluded as checkRunIsIncludedInTable
from make_results_table import algo_labels_dict, algos, metric_names

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 11,
        "font.family": "serif",
        "font.serif": "times",  # because of aweful AISTATS template
        "mathtext.fontset": "cm",
        "text.latex.preamble": (
            r"\usepackage{amssymb} \usepackage{amsmath} "
            r"\usepackage{marvosym} \usepackage{bm}"
        ),
    }
)

MAKE_TIKZ = True

metrics = [NLPD, EXACT_LOSS, RMSE]
optimizer = "L-BFGS-B"

id_list = [str(i) for i in range(5, 9)]  # large scale CPU experiments
id_list = [str(i) for i in range(1, 9)]


def checkRunIsIncluded(algo, run):
    if not checkRunIsIncludedInTable(algo, run):
        return False
    #elif algo == NativeVariationalGPR.get_registry_key() and run.data.tags[NUM_INDUCING_INPUTS] != "1024":
    elif algo == NativeVariationalGPR.get_registry_key() and run.data.tags[NUM_INDUCING_INPUTS] != "2048":
        return False
    elif algo == CGLB.get_registry_key() and run.data.tags[NUM_INDUCING_INPUTS] != "2048":
    #elif algo == CGLB.get_registry_key() and run.data.tags[NUM_INDUCING_INPUTS] != "1024":
        return False
    return True


def get_algo_labels():
    return algo_labels_dict.copy()
algo_labels = get_algo_labels()
inv_map = {v: k for k, v in algo_labels.items()}
#colors = [exact_color()[-1], acgp_color()[-1], cglb_color()[-1], svgp_color()[-1]]
colors = [
    plt.cm.binary(0.8),  # exact
    plt.cm.YlOrRd(0.65), # acgp
    plt.cm.GnBu(0.7),    # cglb
    plt.cm.YlOrRd(0.45), # svgp
]
algo_colors = {algos[i].get_registry_key(): colors[i] for i in range(len(algos))}
symbols = ['x-', '.-', '+-', 'o-']
algo_symbols = {algos[i].get_registry_key(): symbols[i] for i in range(len(algos))}


def get_steps_and_values_from_run(run, metric: str):
    steps, values = get_steps_and_values_from_run_(run_id=run.info.run_id, metric=metric)
    steps = np.array(steps)
    values = np.array(values)
    sorted = np.argsort(steps)
    return steps[sorted], values[sorted]

scaling_factor = 1.2
fig, ax = plt.subplots(figsize=(scaling_factor*4, scaling_factor*2.2), constrained_layout=True)
for exp_id in id_list:
    exp = mlflow.tracking.MlflowClient().get_experiment(exp_id)
    dataset_name = exp.tags[DATASET]
    kernel_name = exp.tags["kernel_name"]

    mlfc = mlflow.tracking.MlflowClient()
    runs = mlfc.search_runs([exp.experiment_id], filter_string="tags.optimizer='%s'" % optimizer)

    for metric in metrics:
        for run in runs:
            if ALGORITHM not in run.data.tags.keys():
                continue  # crashed run
            algo = run.data.tags[ALGORITHM]
            if not checkRunIsIncluded(algo, run):
                continue

            try:
                accepted_steps, metric_values = get_steps_and_values_from_run(run, metric)
                steps_loss, loss_run_times = get_steps_and_values_from_run(run, LOSS_TIME)
                steps_grad, grad_run_times = get_steps_and_values_from_run(run, GRAD_TIME)
                if len(grad_run_times) == len(loss_run_times) - 1:
                    # it can happen that a job is killed just after computing the loss but while computing the gradient
                    assert(steps_grad[-1] == steps_loss[-2])
                    loss_run_times = loss_run_times[:-1]
                setup_time = run.data.metrics[SETUP_TIME]
                run_times = np.cumsum(loss_run_times + grad_run_times) + setup_time

                if len(metric_values) < 5:
                    warnings.warn(f"Run {run.info.run_id} in experiment {run.info.experiment_id} has less than 4 values!")
                    continue
                # we need a -1 because the steps start counting at 1
                ax.plot(run_times[accepted_steps-1], metric_values, color=algo_colors[algo], label=algo_labels[algo],
                        lw=1.5, ls="-", alpha=.7,)
                algo_labels[algo] = None  # to make sure we add the legend entry only once
            except Exception as e:
                warnings.warn(f"{algo} on {dataset_name} using {kernel_name} raised an exception!")
                logging.exception(e)

        algo_labels = get_algo_labels()
        ax.legend(frameon=False)
        leg = ax.get_legend()
        for handle in leg.legendHandles:
            handle.set_alpha(1.)

        ax.ticklabel_format(scilimits=(-3,3))

        ax.set_xlabel(r'time in seconds')
        ax.set_ylabel(r'%s' % metric_names[metric])
        ax.set_xscale('log')
        if MAKE_TIKZ:
            fig.savefig(fname='./output/figures/hyperparametertuning/experiment_4_hyp_tuning' + str(dataset_name) + metric + '.pdf', format='pdf')
        else:
            plt.show()

        plt.cla()
    #stop
