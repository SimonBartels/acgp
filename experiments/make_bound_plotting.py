import argparse
import os
import numpy as np
import pandas as pd
import pathlib
import mlflow
import warnings
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

from utils.result_management.acgp import process_stopped_chol_run
from utils.result_management.cglb import process_cglb_runs
from utils.visualization.save_relative_error import get_log_relative_errors_save
from utils.visualization.visualization_constants import (
    acgp_color,
    cglb_color,
    exact_color,
)

# this import is necessary to set the tracking uri correct
from utils.result_management.result_management import get_steps_and_values_from_run
from utils.result_management.constants import (
    SN2,
    KERNEL,
    DATASET,
    SEED,
    ALGORITHM,
    STEP_TIME,
    SETUP_TIME, EXPERIMENT_TYPE, HYPER_PARAMETER_TUNING_EXPERIMENT,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 11,
        "font.family": "serif",
        "font.serif": "times",  # because of awful AISTATS template
        "mathtext.fontset": "cm",
        "text.latex.preamble": (
            r"\usepackage{amssymb} \usepackage{amsmath} "
            r"\usepackage{marvosym} \usepackage{bm}"
        ),
    }
)

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--use_cache",
    action="store_true",
    help=("Load precomputed results from cache to speed things up."),
)
parser.add_argument("--tight", action="store_true", help="Make narrower plots.")
parser.add_argument("--stacked", action="store_true", help="Make stacked plots.")
args = parser.parse_args()

# Saving plots:
save = True
fig_path = pathlib.Path("output/figures/bounds")
fig_path.mkdir(parents=True, exist_ok=True)

plot_relative_error = False

methods = ["acgp", "cglb", "exact"]
# methods = ["acgp", "exact"]
terms = ["llh", "quad", "log_det"]
# terms = ["quad"]
# terms = ["log_det"]

bounds_path = "./output/results/mlruns"
mlflow.set_tracking_uri(bounds_path)
mlfc = mlflow.tracking.MlflowClient()
experiment_list = [
    exp for exp in mlfc.list_experiments() if exp.experiment_id != "0"
]  # 0 is the default mlflow experiment--it's empty
# experiment_list = [mlflow.get_experiment(e_id) for e_id in ["6", "8", "13"]]   # ids for main paper
# experiment_list = [mlflow.get_experiment(e_id) for e_id in ["13"]]   # ids for main paper
# experiment_list = [mlflow.get_experiment(e_id) for e_id in ["2", "3", "4", "5"]]
# experiment_list = [mlflow.get_experiment(e_id) for e_id in ["8"]]   # ids for main paper
# experiment_list = [mlflow.get_experiment(e_id) for e_id in ["26"]]

seeds = [0, 1, 2, 3, 4]
# seeds = [0]


def process_stopped_chol_runs(runs, sn2, preconditioner=0):
    for r in runs:
        if r.data.tags[ALGORITHM].startswith("MetaCholesky"):
            if int(r.data.tags[ALGORITHM + ".block_size"]) > 20000:
                # this is an exact run
                continue
            if int(r.data.tags["preconditioner_steps"]) != preconditioner:
                continue
            # raise NotImplementedError("I have to exclude the exact runs here!")
            return process_stopped_chol_run(r, sn2)[:7]
    return None


def process_exact_runs(runs):
    for r in runs:
        if r.data.tags[ALGORITHM].startswith("MetaCholesky"):
            if int(r.data.tags[ALGORITHM + ".block_size"]) < 20000:
                # this is an approximate run
                continue
            # below call is insufficient as it essentially ignores the time to build the kernel matrix
            # return r.data.metrics[STEP_TIME]
            _, times = get_steps_and_values_from_run(r.info.run_id, STEP_TIME)
            # there should be only the time to setup the kernel matrix and to run the meta Cholesky
            assert len(times) == 2
            return np.sum(times) + r.data.metrics[SETUP_TIME]
    return None


cglb_bound_label = r"CGLB upper+lower bound"
our_bound_label = r"ACGP upper+lower bound"

show_acgp_for_full_dataset = False


def load_data(exp, method, cache_path):

    if method == "acgp":
        process_func = lambda run: process_stopped_chol_runs(run, float(exp.tags[SN2]))
    else:
        process_func = process_cglb_runs

    path = cache_path / method
    path.mkdir(parents=True, exist_ok=True)

    try:
        if not args.use_cache:
            raise FileNotFoundError

        exact_time = np.load(path / "exact_time.npy")
        num_points = np.load(path / "num_points.npy")
        times = np.load(path / "times.npy")
        log_det_lower_bounds = np.load(path / "log_det_lower_bounds.npy")
        log_det_upper_bounds = np.load(path / "log_det_upper_bounds.npy")
        quad_lower_bounds = np.load(path / "quad_lower_bounds.npy")
        quad_upper_bounds = np.load(path / "quad_upper_bounds.npy")

        if method != "cglb":
            idx = np.load(path / "idx.npy")
        else:
            # CGLB runs have no indices.
            idx = np.array(
                [
                    None,
                ]
                * len(times)
            )

    except FileNotFoundError:
        s = path.parts[-2]
        runs = mlfc.search_runs(
            [exp.experiment_id], filter_string=f"tags.{SEED} = '{s}'"
        )

        (
            idx,
            num_points,
            times,
            log_det_upper_bounds,
            log_det_lower_bounds,
            quad_upper_bounds,
            quad_lower_bounds,
        ) = process_func(runs)

        exact_time = process_exact_runs(runs)

        # log_det_lower_bounds is a mix of arrays and floats. Converting all to float:
        log_det_lower_bounds = [float(v) for v in log_det_lower_bounds]

        if args.use_cache:
            if method != "cglb":
                # CGLB runs have no indices.
                np.save(path / "idx.npy", idx)

            np.save(path / "exact_time.npy", exact_time)
            np.save(path / "num_points.npy", num_points)
            np.save(path / "times.npy", times)
            np.save(path / "log_det_lower_bounds.npy", log_det_lower_bounds)
            np.save(path / "log_det_upper_bounds.npy", log_det_upper_bounds)
            np.save(path / "quad_lower_bounds.npy", quad_lower_bounds)
            np.save(path / "quad_upper_bounds.npy", quad_upper_bounds)

    return (
        idx,
        num_points,
        times,
        exact_time,
        log_det_upper_bounds,
        log_det_lower_bounds,
        quad_upper_bounds,
        quad_lower_bounds,
    )


# Only create the figure once:
# A4 paper:8.3 x 11.7 inches. Template uses margins of 2 * 0.125 in.
if args.tight:
    scaling_factor = 1.4
    fig, ax = plt.subplots(
        figsize=(scaling_factor * 8 / 3, scaling_factor * 8 / 4),
        constrained_layout=True,
    )
elif args.stacked:
    scaling_factor = 1.2
    # For submission:
    # fig, ax = plt.subplots(3, 1, figsize=(scaling_factor*4, scaling_factor*5.5), constrained_layout=True, sharex=True)
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(scaling_factor * 4.5, scaling_factor * 5.5),
        constrained_layout=True,
        sharex=True,
    )
else:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

for i_exp, exp in enumerate(experiment_list):
    if EXPERIMENT_TYPE in exp.tags and HYPER_PARAMETER_TUNING_EXPERIMENT == exp.tags[EXPERIMENT_TYPE]:
        continue  # skip the results of hyper-parameter tuning experiments
    try:
        dataset_name = exp.tags[DATASET]
        kernel_name = exp.tags[KERNEL]
        log_ls2 = exp.tags[KERNEL + ".log_ls2"]
    except:
        # seems to be a crashed experiment
        warnings.warn(f"Crashed experiment? : {exp.name}")
        continue
    sn2 = float(exp.tags[SN2])

    rec_times = {"acgp": [], "cglb": []}
    log_det_uppers = {"acgp": [], "cglb": []}
    log_det_lowers = {"acgp": [], "cglb": []}
    quad_uppers = {"acgp": [], "cglb": []}
    quad_lowers = {"acgp": [], "cglb": []}

    dfs = []

    first_seed = 0
    for s in seeds:

        cache_path = pathlib.Path(os.getcwd())  # pathlib.Path(bounds_path)
        cache_path = (
            cache_path / "cache" / dataset_name / kernel_name / log_ls2 / str(s)
        )
        cache_path.mkdir(parents=True, exist_ok=True)

        # s = int(s)

        # runs = mlfc.search_runs([exp.experiment_id], filter_string=f"tags.{SEED} = '{s}'")

        # exact_time = process_exact_runs(runs)

        # idx, times, log_det_upper_bounds, log_det_lower_bounds, quad_upper_bounds, quad_lower_bounds = process_stopped_chol_runs(runs)
        exact_log_det = 0
        exact_quad = 0
        for method in ["acgp", "cglb"]:
            (
                idx,
                processed_data,
                times,
                exact_time,
                log_det_upper_bounds,
                log_det_lower_bounds,
                quad_upper_bounds,
                quad_lower_bounds,
            ) = load_data(exp, method, cache_path)

            if method == "acgp":
                full_dataset_size = processed_data[-1]
                N = full_dataset_size
                exact_log_det = log_det_upper_bounds[-1]
                exact_quad = quad_upper_bounds[-1]
                # ACGP delivers the exact solution in the last step
                results = {
                    "seed": s,
                    "points": processed_data[-1],
                    "times": exact_time,
                    "log_det_upper": exact_log_det if not plot_relative_error else 0,
                    "quad_upper": exact_quad if not plot_relative_error else 0,
                    "llh_upper": -exact_quad / 2
                    - exact_log_det / 2
                    - N * np.log(2 * np.pi) / 2
                    if not plot_relative_error
                    else 0,
                    "method": "exact",
                }
                dfs.append(pd.DataFrame(results, index=[0]))

            if plot_relative_error:
                log_det_upper_bounds = get_log_relative_errors_save(
                    log_det_upper_bounds, exact_log_det
                )
                log_det_lower_bounds = get_log_relative_errors_save(
                    log_det_lower_bounds, exact_log_det
                )
                quad_upper_bounds = get_log_relative_errors_save(
                    quad_upper_bounds, exact_quad
                )
                quad_lower_bounds = get_log_relative_errors_save(
                    quad_lower_bounds, exact_quad
                )

            results = {
                "seed": [s] * len(times),
                "times": times,
                "points": processed_data,
                "log_det_upper": log_det_upper_bounds,
                "log_det_lower": log_det_lower_bounds,
                "quad_upper": quad_upper_bounds,
                "quad_lower": quad_lower_bounds,
                "llh_lower": (
                    -np.array(log_det_upper_bounds) / 2
                    - np.array(quad_upper_bounds) / 2
                    - N * np.log(2 * np.pi) / 2
                )
                .flatten()
                .tolist(),
                "llh_upper": (
                    -np.array(log_det_lower_bounds) / 2
                    - np.array(quad_lower_bounds) / 2
                    - N * np.log(2 * np.pi) / 2
                )
                .flatten()
                .tolist(),
                "method": [method] * len(times),
            }

            dfs.append(pd.DataFrame(results))

    results = pd.concat(dfs, ignore_index=True)
    results = results.astype({"points": "Int64"})

    for term in terms:

        if args.stacked:
            axes = {
                "quad": ax[0],
                "log_det": ax[1],
                "llh": ax[2],
            }
        else:
            axes = {term: ax}

        scatter_handles = []

        for method in methods:
            subset = results[results["method"] == method]

            color_levels = np.linspace(0.4, 0.8, len(subset.points.unique()))
            scatter_size = 40
            markerwidths = 0.5

            if method == "acgp":
                # cmap = plt.cm.Reds
                colors = acgp_color(len(subset.points.unique()))
                acgp_legend_handle = axes[term].scatter(
                    # [], [], s=60, c=[cmap(color_levels[-2])], label="ACGP bounds"
                    [],
                    [],
                    s=scatter_size,
                    linewidths=markerwidths,
                    c=[colors[-2]],
                    label="ACGP bounds",
                )
            elif method == "cglb":
                # cmap = plt.cm.Blues
                colors = cglb_color(len(subset.points.unique()))
                if len(subset.points.unique()) > 0:
                    cglb_legend_handle = axes[term].scatter(
                        [],
                        [],
                        s=scatter_size,
                        linewidths=markerwidths,
                        c=[colors[-2]],
                        label="CGLB bounds",
                    )
                    # cglb_legend_handle = [axes[term].scatter(
                    #   [], [], s=scatter_size, c=[colors[-2]], label="CGLB bounds",
                    #   marker=marker
                    # )
                    # for marker in ["v", "^"]]
                else:
                    cglb_legend_handle = None

            elif method == "exact":
                # cmap = plt.cm.Greys
                # color_levels = [0.8]
                colors = exact_color(steps=2)[1:]
                # axes[term].axhline(subset[term + "_upper"].iloc[0], c=cmap(color_levels[0]))
                axes[term].axhline(subset[term + "_upper"].iloc[0], c=colors[0])
                exact_legend_handle = mlines.Line2D(
                    [],
                    [],
                    # color=cmap(color_levels[0]),
                    color=colors[0],
                    marker=".",
                    markersize=18,
                    label="Exact GPR",
                )

            # for seed in seeds:
            #    subset = results[results["seed"] == seed]
            #    acgp = subset[subset["method"] == "acgp"]
            #    cglb = subset[subset["method"] == "cglb"]
            #    exact = subset[subset["method"] == "exact"]

            # embed()
            point_groups = subset.groupby("points")

            for p, (points, idx) in enumerate(point_groups.groups.items()):

                if method == "acgp" and points == full_dataset_size:
                    if not show_acgp_for_full_dataset:
                        continue
                # s = point_groups.get_group(points)
                s = subset.loc[idx]
                # embed()

                for bound in ["upper", "lower"]:
                    # TODO: comment back in
                    # assert(len(s["times"]) == len(seeds))
                    assert len(s["times"]) <= len(seeds)
                    if len(s["times"]) < len(seeds):
                        warnings.warn(
                            f"only {len(s['times'])} results for {method} on "
                            f"{dataset_name} using {kernel_name} with log(l)={log_ls2}"
                        )
                    # marker =
                    marker = "^" if bound == "lower" else "v"
                    if method == "exact":
                        marker = "o"
                    scatter_handle = axes[term].scatter(
                        s["times"],
                        s[term + "_" + bound],
                        # s=30 + 40 * p,
                        marker=marker,
                        linewidths=markerwidths,
                        # marker="_",
                        # marker="^" if bound == "lower" else "v",
                        s=scatter_size,
                        # c=[cmap(color_levels[p])],
                        color=colors[p],
                        edgecolors="white",
                        alpha=0.8,
                        # label=f"{method}"
                    )

                    if p == 2 or method == "exact":
                        scatter_handles.append(scatter_handle)

        # Annotate CGLB bounds:
        groups = results.groupby(["method", "points"])
        means = groups.mean()
        stds = groups.std()
        mins = groups.min()
        maxs = groups.max()

        # embed()
        annotations = []
        font_size = 9
        annotation_text = lambda text: f"\\textsf{{\\bfseries {text}}}"

        if "cglb" in methods:
            cglb_means = means.loc["cglb"]
            for p, points in enumerate(cglb_means.index):
                m = cglb_means.loc[points]
                y = mins.loc["cglb", points][term + "_upper"]

                text = axes[term].annotate(
                    text=annotation_text(int(points)),
                    # text,
                    # xy=(m.times, m[term + "_upper"]),
                    xy=(m.times, y),
                    xytext=(0, -18),
                    textcoords="offset points",
                    color=cglb_color(steps=len(cglb_means))[p],
                    fontsize=font_size,
                    horizontalalignment="center",
                    verticalalignment="baseline",
                )
                annotations.append(text)

        acgp_means = means.loc["acgp"]
        if not show_acgp_for_full_dataset:
            acgp_means = acgp_means.drop(index=full_dataset_size)

        for p, points in enumerate(acgp_means.index):
            m = acgp_means.loc[points]
            # y = mins.loc["acgp", points][term + "_lower"]
            y = maxs.loc["acgp", points][term + "_upper"]

            # Get axis coordinates for the location of the text:
            coords_ax = (
                axes[term]
                .transAxes.inverted()
                .transform(axes[term].transData.transform((m.times, y)))
            )
            # Go from axis to data coordinates:
            # a.transData.inverted().transform(a.transAxes.transform((0.05, 0.9)))

            if coords_ax[1] > 0.9:
                # Move label to the right for the first point:
                coordinates = (18, 0)
            else:
                coordinates = (0, 8)
            text = axes[term].annotate(
                text=annotation_text(int(points)),
                # xy=(m.times, m[term + "_lower"]),
                xy=(m.times, y),
                # xytext=(0, -18),
                xytext=coordinates,
                textcoords="offset points",
                color=acgp_color(steps=len(acgp_means))[p],
                fontsize=font_size,
                horizontalalignment="center",
                verticalalignment="baseline",
            )
            annotations.append(text)

        exact_means = means.loc["exact"]
        for p, points in enumerate(exact_means.index):
            m = exact_means.loc[points]
            y = mins.loc["exact", points][term + "_upper"]
            text = axes[term].annotate(
                text=annotation_text(int(points)),
                xy=(m.times, y),
                xytext=(0, -15),
                textcoords="offset points",
                color=exact_color(steps=len(exact_means) + 1)[-1],
                fontfamily="sans-serif",
                fontsize=font_size,
                horizontalalignment="center",
                verticalalignment="baseline",
            )
            annotations.append(text)

        for text in annotations:
            # Set white outline:
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=3, foreground="white", alpha=1),
                    path_effects.Normal(),
                ]
            )

        if "cglb" in methods and not args.stacked:
            axes[term].legend(
                [
                    scatter_handles[4],
                    (scatter_handles[0], scatter_handles[1]),
                    (scatter_handles[2], scatter_handles[3]),
                ],
                ["Exact", "ACGP", "CGLB"],
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=-0.9)},
                fontsize="small",
            )
        elif "cglb" in methods and term == "quad":
            axes[term].legend(
                [
                    scatter_handles[4],
                    (scatter_handles[0], scatter_handles[1]),
                    (scatter_handles[2], scatter_handles[3]),
                ],
                ["Exact", "ACGP", "CGLB"],
                numpoints=1,
                handler_map={tuple: HandlerTuple(ndivide=None, pad=-0.9)},
                fontsize="small",
            )

        latex_dataset_name = dataset_name.replace("wilson_", "")
        if term == "log_det":
            if plot_relative_error:
                ylabel = r"$\mathrm{sign}(r)\log_2 (|r|+1)$"
            else:
                ylabel = r"$\log\,\mathrm{det}\,[\bm{K}]$"
            title = rf"Bounds on the log-determinant term -- \texttt{{{latex_dataset_name}}}"
        elif term == "quad":
            if plot_relative_error:
                ylabel = r"$\mathrm{sign}(r)\log_2 (|r|+1)$"
            else:
                ylabel = r"$\bm{y}^\top \bm{K}^{-1}\bm{y}$"
            title = rf"Bounds on the quadratic term -- \texttt{{{latex_dataset_name}}}"
            # When plotting just the values, a log scale would mess with the distances of upper and lower bound to the exact quantity
            # axes[term].set_yscale("log")
        elif term == "llh":
            if plot_relative_error:
                raise NotImplementedError()
            else:
                ylabel = r"$\log p(\bm{y})$"
            title = rf"Bounds on the marginal log likelihood -- \texttt{{{latex_dataset_name}}}"
            # When plotting just the values, a log scale would mess with the distances of upper and lower bound to the exact quantity
        else:
            raise NotImplementedError("Unknown term")

        axes[term].set_ylabel(ylabel)
        axes[term].ticklabel_format(scilimits=(-3, 3))

        if not args.stacked:
            axes[term].set_xlabel("time in seconds")
            if args.tight:
                title = ""
            axes[term].set_title(title)

            # adjust_text(annotations, only_move={'points':'y', 'texts':'y'})
            # axes[term].legend()

            # plt.tight_layout()
            if save:
                file_postfix = dataset_name + kernel_name + log_ls2
                if args.tight:
                    file_postfix += "_tight"
                fig.savefig(fname=fig_path / f"experiment_4_{term}_{file_postfix}.pdf")
            else:
                plt.show()

            # fig.clear()
            plt.cla()

    if args.stacked:
        ax[2].set_xlabel("time in seconds")

        title = rf"Bounds for \texttt{{{latex_dataset_name}}}"
        # ax[0].set_title(title)

        fig.align_ylabels()

        # adjust_text(annotations, only_move={'points':'y', 'texts':'y'})
        # ax.legend()

        # plt.tight_layout()
        if save:
            file_postfix = dataset_name + kernel_name + log_ls2
            file_postfix += "_stacked"
            fig.savefig(fname=fig_path / f"experiment_4_{file_postfix}.pdf")
        else:
            plt.show()

        # fig.clear()
        # plt.cla()
        for a in ax:
            a.cla()

    #if i_exp == 1:
    #    stop
