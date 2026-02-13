"""
Visualization utilities for A/B test simulation data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def plot_sample_trajectories(
    sim: Dict[str, Any],
    *,
    n_plot: int = 12,
    which: str = "d_obs",          # "d_obs" or "d_true"
    seed: int = 0,
    show_mean: bool = True,
    alpha: float = 0.6,
):
    """
    Plot a handful of experiment trajectories from the simulation output.

    Parameters
    ----------
    sim : dict
        Output of generate_synthetic_ab_data(cfg)
    n_plot : int
        Number of experiments to plot
    which : str
        "d_obs" for noisy observed trajectories, "d_true" for latent true curves
    seed : int
        Random seed for selecting which experiments to plot
    show_mean : bool
        If True, overlay the across-experiment mean trajectory of `which`
    alpha : float
        Line transparency for individual trajectories
    """
    if which not in ("d_obs", "d_true"):
        raise ValueError('which must be "d_obs" or "d_true"')

    Y = sim[which]
    K, T = Y.shape

    rng = np.random.default_rng(seed)
    n_plot = int(min(n_plot, K))
    idx = rng.choice(K, size=n_plot, replace=False)

    t = np.arange(1, T + 1)

    plt.figure()
    for i in idx:
        plt.plot(t, Y[i], alpha=alpha)

    if show_mean:
        plt.plot(t, Y.mean(axis=0), linewidth=2.5, label=f"mean({which})")
        plt.legend()

    plt.xlabel("Day t")
    plt.ylabel(which)
    plt.title(f"Sample trajectories ({which}), n_plot={n_plot}")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_sample_trajectories_aligned_start(
    sim: Dict[str, Any],
    *,
    n_plot: int = 12,
    which: str = "d_obs",          # "d_obs" or "d_true"
    seed: int = 0,
    align_start: bool = True,
    start_value: float = 0.0,      # common starting value after alignment
    show_mean: bool = True,
    alpha: float = 0.6,
):
    """
    Plot sample trajectories, optionally aligning all to start at the same value.

    If align_start=True, each plotted series y_i is transformed to:
        y_i(t) <- y_i(t) - y_i(1) + start_value
    so every trajectory starts at 'start_value' on day 1.
    """
    if which not in ("d_obs", "d_true"):
        raise ValueError('which must be "d_obs" or "d_true"')

    Y = np.asarray(sim[which], float)  # (K,T)
    K, T = Y.shape

    rng = np.random.default_rng(seed)
    n_plot = int(min(n_plot, K))
    idx = rng.choice(K, size=n_plot, replace=False)

    Y_plot = Y[idx].copy()  # (n_plot,T)

    if align_start:
        Y_plot = Y_plot - Y_plot[:, [0]] + start_value  # broadcast day-1 shift

    t = np.arange(1, T + 1)

    plt.figure()
    for j in range(n_plot):
        plt.plot(t, Y_plot[j], alpha=alpha)

    if show_mean:
        mean_traj = Y_plot.mean(axis=0)
        plt.plot(t, mean_traj, linewidth=2.5, label="mean (aligned)" if align_start else "mean")
        plt.legend()

    plt.xlabel("Day t")
    plt.ylabel(f"{which}" + (" (aligned)" if align_start else ""))
    plt.title(f"Sample trajectories ({which}), n_plot={n_plot}, align_start={align_start}")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_last_day_lower_bounds(
    res: dict,
    *,
    T: int,
    true_last_day: float | None = None,
    title: str = "Last-day backward selection: lower bounds vs window start day",
    show_xbar: bool = True,
    ax=None,
):
    """
    Visualize:
      1) noise-only lower bound: xbar_by_k[k] - se_by_k[k]
      2) full lower bound: L_by_k[k]  (already xbar - (eta + cs_pen))
      (+ optional) true last-day ground truth as a horizontal line.

    X-axis is the *start day* of the last-k window:
        start_day(k) = T - k + 1

    A star marks the optimizer-chosen start day (k_star).
    """

    L_by_k = res["L_by_k"]          # dict[int,float]
    xbar_by_k = res["xbar_by_k"]    # dict[int,float]
    se_by_k = res["se_by_k"]        # dict[int,float]  (CS penalty)
    k_star = int(res["k_star"])

    ks = np.array(sorted(L_by_k.keys()), dtype=int)
    if ks.size == 0:
        raise ValueError("res['L_by_k'] is empty.")

    start_days = T - ks + 1

    full_lb = np.array([L_by_k[int(k)] for k in ks], dtype=float)
    xbar = np.array([xbar_by_k[int(k)] for k in ks], dtype=float)
    cs_pen = np.array([se_by_k[int(k)] for k in ks], dtype=float)
    noise_only_lb = xbar - cs_pen

    # sort by start day increasing
    order = np.argsort(start_days)
    start_days = start_days[order]
    ks = ks[order]
    full_lb = full_lb[order]
    noise_only_lb = noise_only_lb[order]
    xbar = xbar[order]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(start_days, noise_only_lb, marker="o",
            label="Lower bound (CS penalty only): x̄ - cs_pen")
    ax.plot(start_days, full_lb, marker="o",
            label="Lower bound (bias+CS): L_k = x̄ - (η_k + cs_pen)")

    if show_xbar:
        ax.plot(start_days, xbar, marker=".", linestyle="--",
                label="x̄ (mean of last-k window)")

    # --- ground-truth last day line ---
    if true_last_day is not None:
        y = float(true_last_day)
        ax.axhline(y, linestyle=":", linewidth=1.5,
                   label=f"Ground truth last day d_T = {y:.4g}")

    # --- star at chosen start day ---
    start_star = T - k_star + 1
    idx_star = np.where(start_days == start_star)[0]
    if idx_star.size > 0:
        j = int(idx_star[0])
        ax.scatter([start_star], [full_lb[j]], marker="*", s=180,
                   label=f"Chosen window start day = {start_star} (k*={k_star})")
        ax.axvline(start_star, linestyle=":", linewidth=1)

    ax.set_xlabel("Start day of last-k window (T-k+1)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax