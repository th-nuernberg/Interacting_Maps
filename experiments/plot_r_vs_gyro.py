"""Plot the network's R (R.txt, rad/s, from --rotationFromTiming) against the gyro (imu.txt).

R is stored as (y, x, z) with R = -(camera angular velocity), so omega = -(R_x, R_y, R_z). The IMU lags the
events by about 4 ms (experiments/v_units.py).

Usage (repository root):
    python/.venv/bin/python experiments/plot_r_vs_gyro.py output/<run>/R.txt --out experiments/results/r_vs_gyro
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GYRO, NETWORK = "#2a78d6", "#eb6834"          # categorical slots 1 and 2


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("r_file", type=Path)
    ap.add_argument("--imu", type=Path, default=Path("res/shapes_rotation/imu.txt"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/r_vs_gyro"))
    ap.add_argument("--lag", type=float, default=0.004, help="IMU lag behind the events (s)")
    ap.add_argument("--window", type=float, nargs=2, default=[20.0, 30.0], help="time window of the axis plot (s)")
    ap.add_argument("--label", default="C++ network, events only (--rotationFromTiming)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    a = np.loadtxt(args.r_file)
    t = a[:, 0]
    omega = -a[:, [2, 1, 3]]
    imu = np.loadtxt(args.imu)
    gyro = np.stack([np.interp(t + args.lag, imu[:, 0], imu[:, 4 + k]) for k in range(3)], -1)

    # Figure 1: the three axes over a window
    sel = (t >= args.window[0]) & (t <= args.window[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, facecolor=SURFACE)
    for k, ax in enumerate(axes):
        style(ax)
        ax.plot(t[sel], gyro[sel, k], color=GYRO, lw=2, label="gyro (imu.txt)")
        ax.plot(t[sel], omega[sel, k], color=NETWORK, lw=2, label=args.label)
        ax.set_ylabel(f"ω_{'xyz'[k]} (rad/s)", color=TEXT_2, fontsize=9)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    axes[-1].set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    fig.suptitle("Camera angular velocity: gyro vs C++ network from event timing", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "omega_vs_gyro_axes.png", dpi=150, facecolor=SURFACE)

    # Figure 2: accuracy over the whole run, 1 s rolling medians (frames with |gyro| > 1 rad/s)
    moving = (np.linalg.norm(gyro, axis=1) > 1.0) & (np.linalg.norm(omega, axis=1) > 0)
    cos = np.sum(omega * gyro, 1) / (np.linalg.norm(omega, axis=1) * np.linalg.norm(gyro, axis=1) + 1e-12)
    ratio = np.linalg.norm(omega, axis=1) / np.maximum(np.linalg.norm(gyro, axis=1), 1e-12)
    centres = np.arange(1.0, t[-1] - 0.5, 0.25)
    roll_cos, roll_ratio = [], []
    for c in centres:
        w = moving & (np.abs(t - c) <= 0.5)
        roll_cos.append(np.median(cos[w]) if w.sum() >= 10 else np.nan)
        roll_ratio.append(np.median(ratio[w]) if w.sum() >= 10 else np.nan)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    style(axes[0])
    axes[0].plot(t, np.linalg.norm(gyro, axis=1), color=GYRO, lw=1.5, label="gyro")
    axes[0].plot(t, np.linalg.norm(omega, axis=1), color=NETWORK, lw=1.5, label="C++ network")
    axes[0].set_ylabel("|ω| (rad/s)", color=TEXT_2, fontsize=9)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    for ax, values, label, ref in [(axes[1], roll_cos, "direction cos\n(1 s median)", 1.0),
                                   (axes[2], roll_ratio, "|ω| ratio net / gyro\n(1 s median)", 1.0)]:
        style(ax)
        ax.axhline(ref, color=TEXT_2, lw=1)
        ax.plot(centres, values, color=NETWORK, lw=2)
        ax.set_ylabel(label, color=TEXT_2, fontsize=9)
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_ylim(0.0, 2.0)
    axes[-1].set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    fig.suptitle("Accuracy of the C++ rotation estimate over the whole recording (events only)", color=TEXT,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "accuracy_over_time.png", dpi=150, facecolor=SURFACE)

    summary = dict(frames=int(moving.sum()), cos_median=float(np.median(cos[moving & (t > 2)])),
                   cos_mean=float(np.mean(cos[moving & (t > 2)])),
                   ratio_median=float(np.median(ratio[moving & (t > 2)])),
                   axis_corr=[float(np.corrcoef(omega[moving, k], gyro[moving, k])[0, 1]) for k in range(3)])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
