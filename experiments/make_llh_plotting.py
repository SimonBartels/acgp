import argparse
import numpy as np
import pickle
import pathlib
import mlflow
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

from utils.mem_efficient_kernels.isotropic_kernel import RBF, OU

# this import is necessary to set the tracking uri correct
from utils.result_management.result_management import get_steps_and_values_from_run
from utils.result_management.constants import (
    SN2,
    EXACT_SOLUTIONS,
    DIAGONAL,
    EQSOL,
    ALGORITHM,
    KERNEL,
    DATASET,
    SEED,
    EXPERIMENT_TYPE,
)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--tight", action="store_true", help="Make narrower plots.")
args = parser.parse_args()

# Saving plots:
save = True
fig_path = pathlib.Path("output/figures/llh")
fig_path.mkdir(parents=True, exist_ok=True)

bounds_path = "./output/results/mlruns"
mlflow.set_tracking_uri(bounds_path)
mlfc = mlflow.tracking.MlflowClient()
experiment_list_ = [
    exp for exp in mlfc.list_experiments() if exp.experiment_id != "0"
]  # 0 is the default mlflow experiment--it's empty

Nsub = 3000
seeds = [0, 1, 2, 3, 4]
# seeds = [0]

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

palette_red = [
    "#9d0208",
    "#d00000",
    "#dc2f02",
    "#e85d04",
    "#f48c06",
    "#faa307",
    "#ffba08",
    "#03071e",
    "#370617",
    "#6a040f",
]
palette_blue = [
    "#012a4a",
    "#013a63",
    "#01497c",
    "#014f86",
    "#2a6f97",
    "#2c7da0",
    "#468faf",
    "#61a5c2",
    "#89c2d9",
    "#a9d6e5",
]
palette_green = [
    "#99e2b4",
    "#88d4ab",
    "#78c6a3",
    "#67b99a",
    "#56ab91",
    "#469d89",
    "#358f80",
    "#248277",
    "#14746f",
    "#036666",
]
palette_pink = [
    "#ea698b",
    "#d55d92",
    "#c05299",
    "#ac46a1",
    "#973aa8",
    "#822faf",
    "#6d23b6",
    "#6411ad",
    "#571089",
    "#47126b",
]
palette_super_red = [
    "#641220",
    "#6e1423",
    "#85182a",
    "#a11d33",
    "#a71e34",
    "#b21e35",
    "#bd1f36",
    "#c71f37",
    "#da1e37",
    "#e01e37",
]

palettes = [palette_red, palette_pink, palette_blue, palette_green]


def process_stopped_chol_runs(runs, sn2, preconditioner=0):
    for r in runs:
        if r.data.tags[ALGORITHM].startswith("MetaCholesky"):
            if int(r.data.tags[ALGORITHM + ".block_size"]) > 20000:
                # this is an exact run
                continue
            if int(r.data.tags["preconditioner_steps"]) != preconditioner:
                continue
            # raise NotImplementedError("I have to exclude the exact runs here!")
            file_name = r.data.tags[EXACT_SOLUTIONS]
            file_name = file_name.split("experiments/")[1]
            d = pickle.load(open(file_name, "rb"))
            diagL = np.squeeze(d[DIAGONAL])
            assert len(diagL.shape) == 1
            alpha = np.squeeze(d[EQSOL])
            return diagL, alpha
    return None


cutout_shade = plt.cm.binary(0.07)
linewidth = 3

# datasets = [ "wilson_kin40k", "protein", "pm25", "metro", "wilson_bike",
#    "wilson_elevators", "wilson_pumadyn32nm", "wilson_pol"]
datasets = [ "wilson_kin40k", "protein", "pm25", "metro", ]
kernels = [RBF(), OU()]

#kernels = [RBF()]
#datasets = ["pm25"]

if args.tight:
    scaling_factor = 1.3
    figsize=(scaling_factor*4, scaling_factor*1.9)
else:
    figsize=(8, 3)

for k in kernels:
    for dataset_name in datasets:
        fig, (ax_zoom, ax) = plt.subplots(
            1,
            2,
            figsize=figsize,
            layout="constrained",
            # layout="compressed",
            # gridspec_kw={"wspace": -0.5, "hspace": 0.2},
        )
        kernel_name = k.name
        experiment_list = [
            e
            for e in experiment_list_
            if not EXPERIMENT_TYPE in e.tags.keys()
            and e.tags[KERNEL] == kernel_name
            and e.tags[DATASET] == dataset_name
        ]
        experiment_list.sort(key=lambda e: float(e.tags[KERNEL + ".log_ls2"]))
        palette_index = 0
        for exp in experiment_list:
            try:
                log_ls2 = exp.tags[KERNEL + ".log_ls2"]
            except:
                # seems to be a crashed experiment
                warnings.warn(f"Crashed experiment? : {exp.name}")
                continue
            sn2 = float(exp.tags[SN2])
            label = f"$\\log\\ell = {int(float(log_ls2))}$"
            for s in seeds:
                runs = mlfc.search_runs(
                    [exp.experiment_id], filter_string=f"tags.{SEED} = '{int(s)}'"
                )
                diagL, alpha = process_stopped_chol_runs(runs, sn2)
                N = diagL.shape[0]

                ax.plot(
                    np.arange(N),
                    -np.cumsum(diagL) - np.cumsum(np.square(alpha)) / 2,
                    linewidth=linewidth,
                    color=palettes[palette_index][s],
                    label=label if s == seeds[-1] else None,
                )
                ax_zoom.plot(
                    np.arange(Nsub),
                    -np.cumsum(diagL[:Nsub]) - np.cumsum(np.square(alpha[:Nsub])) / 2,
                    linewidth=linewidth,
                    color=palettes[palette_index][s],
                )
            palette_index += 1

        ax_zoom.set_facecolor(cutout_shade)
        # ax_zoom.patch.set_alpha(0.9)
        ax.add_patch(
            Rectangle(
                (0, ax_zoom.get_ylim()[1]),
                Nsub,
                ax_zoom.get_ylim()[0],
                facecolor=cutout_shade,
                alpha=0.9,
                edgecolor="black",
            )
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(Nsub, ax_zoom.get_ylim()[1]),
                coordsA=ax.transData,
                xyB=(Nsub, ax_zoom.get_ylim()[1]),
                coordsB=ax_zoom.transData,
            )
        )
        fig.add_artist(
            ConnectionPatch(
                xyA=(Nsub, ax_zoom.get_ylim()[0]),
                coordsA=ax.transData,
                xyB=(Nsub, ax_zoom.get_ylim()[0]),
                coordsB=ax_zoom.transData,
            )
        )
        # print(ax_zoom.get_ylim())
        ax.set_xlabel("Subset size")
        # ax.set_ylabel("marginal log likelihood")
        #ax.set_xlim([0, N])
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="y", scilimits=(-4, 4))

        if args.tight:
            ax.legend(frameon=False, fontsize="small")
        else:
            ax.legend(frameon=False)

        ax_zoom.set_xlabel("Subset size")
        ax_zoom.set_ylabel("$\\log p(\\bm{y})$")
        ax_zoom.set_xlim([0, Nsub])
        ax_zoom.ticklabel_format(axis="y", scilimits=(-4, 4))

        # plt.tight_layout()
        if save:
            file_postfix = dataset_name + kernel_name
            if args.tight:
                file_postfix += "_tight"
            fig.savefig(fname=fig_path / f"llh_{file_postfix}.pdf")
        else:
            plt.show()

            # fig.clear()
        # ax_zoom.cla()
        # ax.cla()
        fig.clf()
        # fig.close()
