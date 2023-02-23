import argparse
import logging
import pathlib
import warnings

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.entities.run_info import RunStatus
import pandas as pd

# this import is necessary to set the tracking uri correct
import utils.result_management.result_management as result_management
from hyperparameter_tuning.utils.gpytorch.models.cglb import CGLB
from hyperparameter_tuning.utils.gpytorch.models.exact_gpr import ExactGPR
from hyperparameter_tuning.utils.gpytorch.models.stopped_cholesky import StoppedCholesky
from hyperparameter_tuning.utils.gpytorch.models.variational_gpr import (
    NativeVariationalGPR,
    OPTIMIZE_INDUCING_INPUTS,
    NUM_INDUCING_INPUTS, SELECTION_SCHEME, RANDOM,
)
from utils.result_management.constants import (
    LOSS_TIME,
    GRAD_TIME,
    EXACT_LOG_DET,
    EXACT_QUAD,
    ALGORITHM,
    RMSE,
    DATASET,
    KERNEL,
    EXACT_LOSS,
    SEED,
    BLOCK_SIZE,
    NLPD,
)
from utils.result_management.result_management import (
    get_steps_and_values_from_run as get_steps_and_values_from_run_,
    get_last_logged_timestamp,
)

metrics = [RMSE, NLPD, EXACT_LOSS]
metric_names = {RMSE: "RMSE", NLPD: "NLPD", EXACT_LOSS: "$-\\log p(\\bm{y})$"}
optimizer = "L-BFGS-B"
# optimizer = "BFGS"  # why is BFGS no good? That's surprising

algos = [ExactGPR, StoppedCholesky, CGLB, NativeVariationalGPR]

algo_keys = {algo.get_registry_key() for algo in algos}

algo_labels_dict = {
    ExactGPR.get_registry_key(): "Exact",
    StoppedCholesky.get_registry_key(): "ACGP",
    CGLB.get_registry_key(): "CGLB",
    NativeVariationalGPR.get_registry_key(): "SVGP",
}

seeds = [str(i) for i in range(5)]


# these git repository versions contain code that was faulty
EXCLUDED_COMMITS = {
    "6a9606b4d21a8706f6ad78c807d873f0f59d2987",  # faulty SVGP implementation
    #"40cb06fc19a06d71b1789f8443a25e2c60d4309c"
    }


def checkRunIsIncluded(algo, run):
    if run.data.tags[SEED] not in seeds:
        # results of a debug run
        return False
    elif algo not in algo_keys:
        return False
    elif RunStatus.from_string(run.info.status) is not RunStatus.FINISHED and \
            (get_last_logged_timestamp(run) - run.info.start_time) / 1000 / 60 / 60 < 11:
        # run not finished and less than 11 hours
        return False
    elif algo == StoppedCholesky.get_registry_key() and run.data.tags[BLOCK_SIZE] != "10240":
        # there are some old results with wrong block size
        return False
    elif algo == StoppedCholesky.get_registry_key() and "r" in run.data.tags.keys():
        # we want to include only the results with the improving optimization scheme
        return False
    elif algo == StoppedCholesky.get_registry_key() and run.data.tags["mlflow.source.git.commit"] != "b618190cc7227e640b39acb37055770dacd67ce2":
        # this is the commit where we significantly improved the bounds!
        # but it made no difference
        return False
    elif algo == CGLB.get_registry_key() and run.data.tags[OPTIMIZE_INDUCING_INPUTS] == "False":
        # optimizing inducing inputs gives better results with little overhead
        return False
    elif SELECTION_SCHEME in run.data.tags and run.data.tags[SELECTION_SCHEME] == RANDOM:
        return False
    elif run.data.tags["mlflow.source.git.commit"] in EXCLUDED_COMMITS:
        # exclude experiments containing mistake in SGPR implementation!
        return False
    return True

def get_steps_and_values_from_run(run, metric: str):
    steps, values = get_steps_and_values_from_run_(
        run_id=run.info.run_id, metric=metric
    )
    steps = np.array(steps)
    values = np.array(values)
    sorted = np.argsort(steps)
    return steps[sorted], values[sorted]


def load_results(fname, force_recompute=True):
    if not fname.exists() or force_recompute:
        if "gpu" in fname.name:
            id_list = [str(i) for i in range(1, 5)]  # GPU experiments
            # id_list = ["6"]
        else:
            id_list = [str(i) for i in range(5, 9)]  # CPU experiments

        ex_df_list = []
        for exp_id in id_list:
            result_lists = dict()
            exp = mlflow.tracking.MlflowClient().get_experiment(exp_id)
            dataset_name = exp.tags[DATASET]
            if "wilson" in dataset_name:
                dataset_name = dataset_name.split("_")[-1]
            kernel_name = exp.tags["kernel_name"]

            mlfc = mlflow.tracking.MlflowClient()
            runs = mlfc.search_runs(
                [exp.experiment_id], filter_string=f"tags.optimizer='{optimizer}'"
            )

            for run in runs:
                try:
                    algo = run.data.tags[ALGORITHM]
                except KeyError:
                    continue

                if not checkRunIsIncluded(algo, run):
                    continue

                algo_name = algo_labels_dict[algo]
                if algo == NativeVariationalGPR.get_registry_key() or algo == CGLB.get_registry_key():
                    algo_name = (
                        algo_name + " (" + run.data.tags[NUM_INDUCING_INPUTS] + ")"
                    )
                if algo_name not in result_lists.keys():
                    result_lists[algo_name] = {m: [] for m in metrics}
                try:
                    for m in metrics:
                        _, values = get_steps_and_values_from_run(run, m)
                        result_lists[algo_name][m].append(values[-1])
                except Exception as e:
                    warnings.warn(f"Fetching for results for {algo_name} on {dataset_name} using {kernel_name} caused an exception.")
                    logging.exception(e)

            for ex, ex_dict in result_lists.items():
                lengths = [len(v) for v in ex_dict.values()]
                ml = np.max(lengths)
                for metric, values in ex_dict.items():
                    if len(values) < ml:
                        ex_dict[metric] = values + [
                            None,
                        ] * (ml - len(values))

                ex_df = pd.DataFrame(ex_dict)
                ex_df["model"] = ex
                ex_df["dataset"] = dataset_name
                ex_df["kernel"] = kernel_name
                ex_df_list.append(ex_df)

        df = pd.concat(ex_df_list, ignore_index=True)
        df.to_csv(fname, index=False)
    else:
        df = pd.read_csv(fname, index_col=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--recompute",
        action="store_true",
        default=True,
        help="Recompute values from MLFlow files.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=pathlib.Path,
        default="output/tex",
        help="Directory to store TeX output and auxiliary files.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["cpu", "gpu", "both"],
        default="both",
        help="The type of hardware to produce the results table for.",
    )
    args = parser.parse_args()

    # Create output directory:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "both":
        hardware = ["cpu", "gpu"]
    else:
        hardware = [args.type]

    for hw in hardware:
        df = load_results(
            args.output_dir / f"results_{hw}.csv", force_recompute=args.recompute
        )

        lower_is_better = "$\\downarrow$"
        higher_is_better = "$\\uparrow$"

        if hw == "cpu":
            datasets = ["metro", "pm25", "kin40k", "protein"]
            dataset_labels = ["\\METRO", "\\PM", "\\KINFORTYK", "\\PROTEIN"]
        elif hw == "gpu":
            # pole dataset is named wrongly:
            df = df.replace({"dataset": "pol"}, "pole")
            datasets = ["bike", "pole", "elevators", "pumadyn32nm"]
            dataset_labels = ["\\BIKE", "\\POLETELECOMM", "\\ELEVATORS", "\\PUMADYN"]

        models = ["Exact", "ACGP", "CGLB (1024)", "CGLB (2048)", "CGLB (4096)", "SVGP (1024)", "SVGP (2048)", "SVGP (4096)"]
        metrics_descr = [
            (RMSE, "RMSE", lower_is_better),
            (NLPD, "NLPD", lower_is_better),
            (EXACT_LOSS, r"$\log p(\vy)$", higher_is_better),
        ]

        # Let's report the positive log p(y):
        df["EXACT_LOSS"] *= -1

        tex_out = (
            "\sisetup{\n"
            "    separate-uncertainty,\n"
            "    retain-zero-uncertainty,\n"
            # "    round-precision = 3,\n"
            # "    round-mode = uncertainty,\n"
            "    drop-exponent = true,\n"
            "    exponent-mode = fixed,\n"
            "    text-series-to-math = true,\n"
            "    detect-all = true,\n"
            "}\n"
        )

        # Precisions for printing the table:
        if hw == "cpu":
            precisions = [4, 3, 0]
            fixed_exponents = [-2, -1, 4]
            table_format = ["2.2", "2.2", "1.4"]
        elif hw == "gpu":
            precisions = [4, 3, 0]
            fixed_exponents = [-2, -1, 4]
            table_format = ["2.2", "2.2", "1.4"]

        tex_out += (
            "\\begin{tabular}{"
            "ll\n"
            f"S[table-format = {table_format[0]}(4), round-precision=2, fixed-exponent={fixed_exponents[0]}]\n"
            f"S[table-format = +{table_format[1]}(6), round-precision=3, fixed-exponent={fixed_exponents[1]}]\n"
            f"S[table-format = +{table_format[2]}(5), round-precision=3, fixed-exponent={fixed_exponents[2]}]\n"
            "}\n"
        )
        tex_out += r"\toprule"
        # tex_out += "{Dataset} & {Model} "
        tex_out += "{\\bfseries Dataset} & {\\bfseries Model} "
        for m, (metric_key, metric, up_or_down) in enumerate(metrics_descr):
            tex_out += (
                f" & {{\\bfseries {metric} / $10^{{{fixed_exponents[m]}}}$ "
                f"({up_or_down})}}"
            )
        tex_out += "\\\\\n"

        for dataset, label in zip(datasets, dataset_labels):
            tex_out += "\\midrule\n"
            tex_out += f"\\multirow{{{len(models)}}}{{*}}{{{label}}}"

            # Get results for this dataset and group by model:
            subset = df[df.dataset == dataset]
            model_metrics = subset.groupby("model")

            # Means and stds per model:
            means = model_metrics.mean()
            stds = model_metrics.std(ddof=0)

            # Find the best models, excluding Exact (since it's the gold standard):
            model_means = means.loc[models[1:]]
            best_model = {
                metric_key: model_means[metric_key].idxmin()
                if metric_type == lower_is_better
                else model_means[metric_key].idxmax()
                for metric_key, _, metric_type in metrics_descr
            }

            # Check the maximum number of metric values for each model (i.e., number of
            # completed runs):
            completed_runs = model_metrics.count().max("columns")
            wrong_number_of_runs = ", ".join(
                [
                    f"{m}: {completed_runs[m]}"
                    for m in completed_runs.index[completed_runs != len(seeds)]
                ]
            )
            if not all(completed_runs <= len(seeds)):
                warnings.warn(
                    f"There are {len(seeds)} seeds for dataset {dataset}, but the "
                    "following models have a different amount of metrics: "
                    f"{wrong_number_of_runs}"
                )

            for model in models:

                if model == "Exact":
                    tex_out += f" & \\goldstandard {model} "
                else:
                    tex_out += f" & {model} "
                for m, (metric_key, metric, _) in enumerate(metrics_descr):

                    mean = means.loc[model][metric_key]
                    std = stds.loc[model][metric_key]

                    this_model_is_the_best = model == best_model[metric_key]

                    if np.isnan(mean) or np.isnan(std):
                        tex_out += " & {NA}"
                    else:
                        p = precisions[m]
                        #if p < 0:
                        #    mean *= 10**p
                        #    std *= 10**p
                        #    p = 0
                        # Funky format for siunitx:
                        if this_model_is_the_best:
                            # tex_out += f" & \\bfseries {mean:010.6f} \pm {std:.6f}"
                            tex_out += (
                                f" & \\bfseries {mean:10.{p}f}({std * 10**p:.0f})"
                            )

                        elif model == "Exact":
                            # tex_out += f" & \\goldstandard {mean:010.6f} \pm {std:.6f}"
                            tex_out += (
                                f" & \\goldstandard {mean:10.{p}f}({std * 10**p:.0f})"
                            )

                        else:
                            # tex_out += f" & {mean:010.4f} \pm {std:.6f}"
                            tex_out += f" & {mean:10.{p}f}({std * 10**p:.0f})"

                tex_out += "\\\\\n"

        tex_out += "\\bottomrule\n"
        tex_out += "\\end{tabular}"

        with open(args.output_dir / f"results_table_{hw}.tex", "w") as f:
            f.write(tex_out)

    # TODO: run that script automatically after result collection ...
    warnings.warn(
        "If the negative MLL values are missing, it might be necessary to run "
        "experiments.experiments_simon.local_auxilary_computations.py --- "
        "gosh code gets messy after a while"
    )
