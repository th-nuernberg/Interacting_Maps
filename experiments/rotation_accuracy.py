"""Score runs of the C++ network against the IMU (shapes_rotation): R against the gyro, R's delay, and G's polarity
agreement.

R.txt holds R = -(angular velocity) as (y, x, z) in rad/s; the gyro (imu.txt, camera frame) lags the events by
4 ms. Frames with |gyro| > 1 rad/s after 2 s are scored. The delay is the shift of the gyro that best matches the
whole vector (least squares). G's polarity agreement needs --dumpEvents (event_diagnostics.bin).

Usage (repository root):
    python/.venv/bin/python experiments/rotation_accuracy.py "default=output/ev_base" "both=output/ev_both" ...
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from polarity_check import DIAGNOSTICS, sign_check  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GYRO, SERIES = "#2a78d6", ["#eb6834", "#1baf7a", "#8a5cd1", "#d1a21b"]
LAG = 0.004


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def gyro_at(imu, t, shift=0.0):
    return np.stack([np.interp(t + LAG + shift, imu[:, 0], imu[:, 4 + k]) for k in range(3)], -1)


def score(run, imu):
    a = np.loadtxt(run / "R.txt")
    t, omega = a[:, 0], -a[:, [2, 1, 3]]
    g = gyro_at(imu, t)
    m = (np.linalg.norm(g, axis=1) > 1) & (t > 2)
    cos = np.sum(omega * g, 1) / (np.linalg.norm(omega, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
    ratio = np.linalg.norm(omega, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-12)
    err = np.linalg.norm(omega - g, axis=1)
    shifts = np.arange(-0.04, 0.0201, 0.001)
    delay = -shifts[int(np.argmin([np.mean(np.linalg.norm(omega[m] - gyro_at(imu, t[m], s), axis=1) ** 2)
                                   for s in shifts]))]
    acc = np.linalg.norm(np.gradient(g, t, axis=0), axis=1)
    r = dict(cos_median=float(np.median(cos[m])), cos_mean=float(np.mean(cos[m])),
             spikes=float(np.mean(cos[m] < 0.9)), spikes_05=float(np.mean(cos[m] < 0.5)),
             ratio_median=float(np.median(ratio[m])), ratio_q=[float(np.percentile(ratio[m], q)) for q in (25, 75)],
             err_rms=float(np.sqrt(np.mean(err[m] ** 2))), err_median=float(np.median(err[m])),
             corr=[float(np.corrcoef(omega[m, k], g[m, k])[0, 1]) for k in range(3)], delay_ms=float(delay * 1e3),
             spikes_high_acc=float(np.mean(cos[m & (acc > 100)] < 0.9)))
    dump = run / "event_diagnostics.bin"
    if dump.exists():
        d = np.fromfile(dump, dtype=DIAGNOSTICS)
        agree, ok = sign_check(d, d["MRgyro"].astype(np.float64))
        r["polarity_agreement"] = float(agree.sum() / ok.sum())
    return r, (t, omega, g, cos, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=output/<run>")
    ap.add_argument("--imu", type=Path, default=Path("res/shapes_rotation/imu.txt"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/rotation_accuracy"))
    ap.add_argument("--plot", nargs="*", default=None, help="labels to plot (default: first and last)")
    ap.add_argument("--window", type=float, nargs=2, default=[24.0, 26.0])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    imu = np.loadtxt(args.imu)

    results, series = {}, {}
    for label, path in (r.split("=", 1) for r in args.runs):
        results[label], series[label] = score(Path(path), imu)
        r = results[label]
        print(f"{label:28s} cos {r['cos_median']:.4f} / {r['cos_mean']:.4f}  cos<0.9 {r['spikes']:6.2%} "
              f"(at >100 rad/s² {r['spikes_high_acc']:6.2%})  ratio {r['ratio_median']:.3f} "
              f"[{r['ratio_q'][0]:.2f}, {r['ratio_q'][1]:.2f}]  |Δω| rms {r['err_rms']:.3f}  "
              f"corr {np.round(r['corr'], 3)}  delay {r['delay_ms']:+.0f} ms"
              + (f"  G polarity {r['polarity_agreement']:.3f}" if "polarity_agreement" in r else ""))
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))

    labels = args.plot or [list(results)[0], list(results)[-1]]
    # Figure 1: the three axes in a window with fast changes
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, facecolor=SURFACE)
    for k, ax in enumerate(axes):
        style(ax)
        t, _, g, _, _ = series[labels[0]]
        sel = (t >= args.window[0]) & (t <= args.window[1])
        ax.plot(t[sel], g[sel, k], color=GYRO, lw=2.2, label="gyro (IMU)")
        for colour, label in zip(SERIES, labels):
            t, omega, _, _, _ = series[label]
            sel = (t >= args.window[0]) & (t <= args.window[1])
            ax.plot(t[sel], omega[sel, k], color=colour, lw=1.6, label=label)
        ax.set_ylabel(f"ω_{'xyz'[k]} (rad/s)", color=TEXT_2, fontsize=9)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1 + len(labels), frameon=False, fontsize=9,
                   labelcolor=TEXT)
    axes[-1].set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    fig.suptitle("Angular velocity: network (events only) against the IMU", color=TEXT, fontsize=11, x=0.01,
                 ha="left", y=0.995)
    fig.tight_layout()
    fig.savefig(args.out / "omega_window.png", dpi=150, facecolor=SURFACE)

    # Figure 2: share of frames with cos < 0.9 by angular acceleration
    edges = [0, 20, 50, 100, 200, 1e9]
    names = ["0–20", "20–50", "50–100", "100–200", "> 200"]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    width = 0.8 / len(labels)
    for i, (colour, label) in enumerate(zip(SERIES, labels)):
        t, _, g, cos, m = series[label]
        acc = np.linalg.norm(np.gradient(g, t, axis=0), axis=1)
        vals = [np.mean(cos[m & (acc >= lo) & (acc < hi)] < 0.9) for lo, hi in zip(edges[:-1], edges[1:])]
        x = np.arange(len(names)) + (i - (len(labels) - 1) / 2) * width
        ax.bar(x, vals, width * 0.9, color=colour, label=label)
    ax.set_xticks(range(len(names)), names)
    ax.set_xlabel("angular acceleration of the camera (rad/s²)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("frames with cos < 0.9", color=TEXT_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("Where R goes wrong: fast changes of the rotation", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "spikes_by_acceleration.png", dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
