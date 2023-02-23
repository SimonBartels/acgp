import argparse
import pathlib

import numpy as np
import random
import numpy.random
import matplotlib.pyplot as plt
import scipy.linalg

from acgp.blas_wrappers.openblas.openblas_wrapper import OpenBlasWrapper
from acgp.bound_computation import Bounds
from acgp.hooks.stop_hook import StopHook
from utils.kernel_evaluator import get_kernel_evaluator
from acgp.meta_cholesky import MetaCholesky
from utils.mem_efficient_kernels.isotropic_kernel import RBF

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

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--use_cache",
    action="store_true",
    help=("Load precomputed results from cache to speed things up."),
)
parser.add_argument(
    "--cache_path",
    type=pathlib.Path,
    default="./cache/toy_example",
    help="Where to store the cached files.",
)
parser.add_argument(
    "--fig_path",
    type=pathlib.Path,
    default="./output/figures/toy",
    help="Where to save the plots.",
)
args = parser.parse_args()

args.fig_path.mkdir(parents=True, exist_ok=True)
if args.use_cache:
    args.cache_path.mkdir(parents=True, exist_ok=True)

# set seeds
seeds = list(range(10))

# Define properties:
sn2 = 1e-1
k = RBF().initialize(log_ls2=-2.0, log_var=0.0)

N_func = 100
Nmax = 10000  # maximal amount of datapoints we are willing to process
N = 10**12  # actual size of the dataset

Xs = np.linspace(0, 1, 100).reshape([-1, 1])  # inputs that we will visualize f on

relative_errors = []
selected_subset_sizes = []
ground_truths = []

for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")
    try:

        if not args.use_cache:
            raise FileNotFoundError

        X = np.load(args.cache_path / f"X_{N}_{seed}.npy")
        y = np.load(args.cache_path / f"y_{N}_{seed}.npy")
        Xf = np.load(args.cache_path / f"Xf_{N_func}_{seed}.npy")
        true_f = np.load(args.cache_path / f"true_f_{N_func}_{seed}.npy")
        y_acgp = np.load(args.cache_path / f"y_acgp_{N_func}_{seed}.npy")
        var_acgp = np.load(args.cache_path / f"var_acgp_{N_func}_{seed}.npy")
        M = np.load(args.cache_path / f"M_{N}_{seed}.npy")

    except FileNotFoundError:
        # prepare variables
        r = 0.01
        a = 0.0

        block_size = 1000

        # build true function
        Xf = np.linspace(0, 1, N_func).reshape([-1, 1])
        Lf = np.linalg.cholesky(k.K(Xf) + 1e-9 * np.eye(Xf.shape[0]))
        y_ = np.random.randn(Xf.shape[0], 1)
        alpha_f = scipy.linalg.solve_triangular(Lf.T, y_, lower=False)
        true_f = k.K(Xs, Xf) @ alpha_f

        # sample dataset
        X = np.random.rand(Nmax, 1)
        y = k.K(X, Xf) @ alpha_f + np.sqrt(sn2) * np.random.randn(Nmax, 1)
        #X = np.random.rand(15000, 1)[:Nmax]
        #y = k.K(X, Xf) @ alpha_f + np.sqrt(sn2) * np.random.randn(15000, 1)[:Nmax]

        # compute ground-truth for Nmax
        L = np.linalg.cholesky(k.K(X) + sn2 * np.eye(X.shape[0]))
        alpha = scipy.linalg.solve_triangular(L, y, lower=True)
        llh = (
            -np.sum(np.log(np.diag(L)))
            - np.sum(np.square(alpha)) / 2
            - Nmax / 2 * np.log(2 * np.pi)
        )
        alpha_pred = scipy.linalg.solve_triangular(L.T, alpha)

        # initialize ACGP
        A = np.zeros([Nmax, Nmax], order="F")
        z = np.asfortranarray(y.copy())
        chol = MetaCholesky(
            block_size=block_size,
            initial_block_size=block_size,
            blaswrapper=OpenBlasWrapper(),
        )
        hook = StopHook(N=N, min_noise=sn2, relative_tolerance=r, absolute_tolerance=a)
        chol.run_configuration(
            A,
            z,
            kernel_evaluator=get_kernel_evaluator(X=X, k=lambda *args: k.K(*args), sn2=sn2),
            hook=hook,
        )

        #print(f"Finished at iteration {hook.iteration}")
        # read out bounds
        log_det_lower_bounds = [l for _, l, _, _ in hook.bound_estimates]
        log_det_upper_bounds = [u for u, _, _, _ in hook.bound_estimates]
        quad_lower_bounds = [l for _, _, _, l in hook.bound_estimates]
        quad_upper_bounds = [u for _, _, u, _ in hook.bound_estimates]
        llh_lower = (
            -np.array(log_det_upper_bounds) / 2
            - np.array(quad_upper_bounds) / 2
            - N * np.log(2 * np.pi) / 2
        ).flatten()  # .tolist()
        llh_upper = (
            -np.array(log_det_lower_bounds) / 2
            - np.array(quad_lower_bounds) / 2
            - N * np.log(2 * np.pi) / 2
        ).flatten()  # .tolist()
        achievable_errors = (
            (llh_upper - llh_lower)
            / np.min(np.abs(np.stack([llh_upper, llh_lower])), axis=0)
            / 2
        )
        # print(achievable_errors)
        # print(llh_upper - llh_lower)
        # print(hook.finished)

        estimate = llh_lower[-1] / 2 + llh_upper[-1] / 2

        if not hook.finished:
            bounds = Bounds(delta=0.0, N=Nmax, min_noise=sn2)
            t0 = hook.iteration
            t = min(Nmax, hook.iteration + block_size)
            U_det, L_det, U_quad, L_quad = bounds.get_bound_estimators(
                t0=t0,
                log_sub_det=hook.ldet,
                sub_quad=hook.quad,
                A_diag=np.diag(A[t0:t, t0:t]).reshape(-1, 1),
                A_diag_off=np.diag(A[t0:t, t0:t], -1).reshape(-1, 1),
                noise_diag=np.array(sn2).reshape([1]),
                y=z[t0:t, :],
            )
            Nmax_estimate = (
                -(U_det + L_det) / 2 / 2
                - (U_quad + L_quad) / 2 / 2
                - Nmax / 2 * np.log(2 * np.pi)
            )
            print(f"relative error on NMax llh: {np.abs((Nmax_estimate - llh) / llh)}")
            relative_errors.append(np.abs((Nmax_estimate - llh) / llh))
            ground_truths.append(llh)

        s = hook.iteration
        if not hook.finished:
            # execute step 3 of the Cholesky if necessary
            M = s + min(block_size, Nmax - s)
            A[s:M, s:M] = np.linalg.cholesky(A[s:M, s:M])
            z[s:M, :] = scipy.linalg.solve_triangular(np.tril(A[s:M, s:M]), z[s:M, :], lower=True)
        else:
            M = Nmax

        selected_subset_sizes.append(M)

        print(f"Processed {M} data out of {N:g} ({M/N*100:g}%).")

        # Compute predictive mean and variance of ACGP:
        alpha_hat = scipy.linalg.solve_triangular(np.tril(A[:M, :M]).T, z[:M, :])
        ks = k.K(Xs, X[:M, :])
        y_acgp = ks @ alpha_hat
        v = scipy.linalg.solve_triangular(np.tril(A[:M, :M]), ks.T, lower=True)
        var_acgp = k.Kdiag(Xs) - np.sum(np.square(v), axis=0)

        if args.use_cache:
            np.save(args.cache_path / f"X_{N}_{seed}.npy", X)
            np.save(args.cache_path / f"y_{N}_{seed}.npy", y)
            np.save(args.cache_path / f"Xf_{N_func}_{seed}.npy", Xf)
            np.save(args.cache_path / f"true_f_{N_func}_{seed}.npy", true_f)
            np.save(args.cache_path / f"y_acgp_{N_func}_{seed}.npy", y_acgp)
            np.save(args.cache_path / f"var_acgp_{N_func}_{seed}.npy", var_acgp)
            np.save(args.cache_path / f"M_{N}_{seed}.npy", M)

    # ==================== Make plot ====================

    fig, ax = plt.subplots(figsize=(4, 2.5), constrained_layout=True)

    colors = {
        "data": plt.cm.binary(0.45),
        "true": plt.cm.binary(0.8),
        "acgp": plt.cm.YlOrRd(0.75), # .65
    }


    #Xf = Xf.squeeze()
    #Xs = Xs.squeeze()
    y_acgp = y_acgp.squeeze()
    s_acgp = np.sqrt(var_acgp) + np.sqrt(sn2)

    # Let's only plot a subset of the data to avoid overly complex files:
    #idx = np.random.permutation(Nmax)#[:3000]  # choose a random subset of 3000
    idx = np.arange(M)
    ax.scatter(
        X[idx, 0], y[idx, 0], s=5, marker=".", color=colors["data"], alpha=0.7, label="Data"
    )

    ax.plot(Xs.squeeze(), true_f, "--", color=colors["true"], label="True function")

    ax.fill_between(
        Xs.squeeze(),
        y_acgp - 2 * s_acgp,
        y_acgp + 2 * s_acgp,
        color=colors["acgp"],
        ec="none",
        alpha=0.15,
    )
    ax.plot(Xs, y_acgp, color=colors["acgp"], label="ACGP", alpha=0.7)

    ax.legend(scatterpoints=3, fontsize="small")#, frameon=False)
    ax.set_xlim(0, 1)


    fig.savefig(fname=args.fig_path / f"toy__seed_{seed}.pdf")

if not args.use_cache:
    print(f"Ground-truths over {len(seeds)} seeds: "
            f"{np.mean(ground_truths)} +/- {np.std(ground_truths)}")

    print(f"Relative errors over {len(seeds)} seeds: "
            f"{np.mean(relative_errors)} +/- {np.std(relative_errors)}")

    print(f"Selected subset sizes: {np.mean(selected_subset_sizes)} +/- "
            f"{np.std(selected_subset_sizes)}")

