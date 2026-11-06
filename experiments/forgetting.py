"""How much error does each event add to I, and how fast must I forget? (shapes_rotation)

Without decay, I at a pixel is the sum of its event steps. If every event carries an independent error of variance
sigma^2 (in steps^2: threshold noise, pixel mismatch, noise events), the spread of I over pixels grows linearly with
the pixel's number of events N, Var(I) ~ a + sigma^2 N. The true log intensity stays bounded (a: scene contrast in
steps). A wrong step ratio per pixel would add a term growing as N^2. This script integrates the events exactly like
the C++ network without decay (I in units of c_OFF, ON step = c_ON / c_OFF; equal, global or per pixel, causal) and
fits that model.

Forgetting rule derived from it: an exponential leak with time constant tau keeps sigma^2 * rate * tau / 2 of error
variance per pixel, so the difference of two neighbours (what G reads) carries sigma^2 * rate * tau. Forgetting
before that reaches one step^2 gives tau* = 1 / (sigma^2 * rate), with rate the pixel's event rate.

The second part checks tau* against C++ runs that sweep --decayTime (with --dumpEvents 20): polarity agreement
sign(-G . u_gyro) = p as a function of tau, for equal and per-pixel contrast steps.

Usage (repository root):
    for m in equal pixel; do for tau in 0.01 0.03 0.06 0.1 0.3 1.0; do
        ./InteractingMaps --startTime 0 --endTime 60 --rotationFromTiming true --writeFrames false --dumpEvents 20 \
            --contrastSteps $m --decayTime $tau --resultsDirectory tau_${m}_${tau}; done; done
    python/.venv/bin/python experiments/forgetting.py --sweep 0.01 0.03 0.06 0.1 0.3 1.0
"""
import sys
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
WIDTH, HEIGHT = 240, 180


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def integrate(pix, p, mode, prior=100.0):
    """I after every event (per pixel, no decay) and the pixel's event count N, in event order."""
    on, off = p > 0, p < 0
    on_before = np.cumsum(on) - on                     # global counts before each event (causal)
    off_before = np.cumsum(off) - off
    global_ratio = (off_before + 1.0) / (on_before + 1.0)
    order = np.argsort(pix, kind="stable")             # per pixel, time order kept
    ps, pixs = p[order], pix[order]
    start = np.r_[True, pixs[1:] != pixs[:-1]]
    group_start = np.maximum.accumulate(np.where(start, np.arange(len(ps)), 0))
    n_before = np.arange(len(ps)) - group_start        # events of this pixel before this one
    if mode == "equal":
        ratio = np.ones(len(ps))
    elif mode == "global":
        ratio = global_ratio[order]
    else:                                               # per pixel, shrunk towards the global ratio (C++ "pixel")
        on_s = (ps > 0).astype(np.float64)
        before = np.cumsum(on_s) - on_s                 # ON events before, over all pixels in this order
        c_on_before = before - before[group_start]      # ... of this pixel
        c_off_before = n_before - c_on_before
        ratio = (c_off_before + prior * global_ratio[order]) / (c_on_before + prior)
    step = np.where(ps > 0, ratio, -1.0)
    csum = np.cumsum(step)
    I = csum - (csum - step)[group_start]              # running sum within the pixel
    return I, n_before + 1, order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("experiments/results/v_units/events_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/forgetting"))
    ap.add_argument("--sweep", type=str, nargs="*", default=[], help="decay times of output/tau_<mode>_<tau> runs")
    ap.add_argument("--fit-range", type=float, nargs=2, default=[20, 2000], help="N range of the fit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    c = np.load(args.cache)
    t, x, y, p = c["t"], c["x"].astype(np.int64), c["y"].astype(np.int64), c["p"].astype(np.int8)
    pix = y * WIDTH + x
    duration = t[-1] - t[0]
    counts = np.bincount(pix, minlength=WIDTH * HEIGHT)
    active = counts > 200
    rate = counts[active] / duration
    print(f"{len(t)} events over {duration:.1f} s; event rate of active pixels: median {np.median(rate):.1f}/s, "
          f"IQR {np.percentile(rate, 25):.1f}-{np.percentile(rate, 75):.1f}/s")

    edges = np.unique(np.round(np.logspace(0, np.log10(4000), 40)).astype(int))
    results = {}
    for mode in ["equal", "global", "pixel"]:
        I, N, _ = integrate(pix, p, mode)
        b = np.digitize(N, edges) - 1
        centres, var, mean = [], [], []
        for k in range(len(edges) - 1):
            sel = b == k
            if sel.sum() < 2000:
                continue
            centres.append(np.sqrt(edges[k] * edges[k + 1]))
            var.append(float(np.var(I[sel])))
            mean.append(float(np.mean(I[sel])))
        centres, var = np.array(centres), np.array(var)
        fit = (centres >= args.fit_range[0]) & (centres <= args.fit_range[1])
        # Var = a + s2 N + q N^2, least squares with relative weights (variance spans decades)
        A = np.stack([np.ones(fit.sum()), centres[fit], centres[fit] ** 2], 1) / var[fit, None]
        a, s2, q = np.linalg.lstsq(A, np.ones(fit.sum()), rcond=None)[0]
        results[mode] = dict(N=centres.tolist(), var=var.tolist(), mean=mean, a=float(a), sigma2=float(s2), q=float(q))
        print(f"{mode:6s}: Var(I) = {a:7.2f} + {s2:.4f} N + {q:.2e} N^2   (sigma = {np.sqrt(max(s2, 0)):.3f} steps; "
              f"at N = 1000: linear part {s2 * 1000:.0f}, quadratic part {q * 1e6:.0f})")

    s2 = results["pixel"]["sigma2"]
    tau = 1.0 / (s2 * rate)
    print(f"tau* = 1 / (sigma^2 rate) with sigma^2 = {s2:.4f}: median {np.median(tau):.2f} s, "
          f"IQR {np.percentile(tau, 25):.2f}-{np.percentile(tau, 75):.2f} s (inherited: 0.1 s)")
    results["rate"] = dict(median=float(np.median(rate)), q25=float(np.percentile(rate, 25)), q75=float(np.percentile(rate, 75)))
    results["tau_star"] = dict(median=float(np.median(tau)), q25=float(np.percentile(tau, 25)), q75=float(np.percentile(tau, 75)))

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style(ax)
    for colour, mode in zip(SERIES, ["equal", "global", "pixel"]):
        r = results[mode]
        N = np.array(r["N"])
        ax.plot(N, r["var"], "o", color=colour, ms=4, label=f"{mode} steps")
        Nf = N[(N >= args.fit_range[0]) & (N <= args.fit_range[1])]   # the model only where it was fitted
        ax.plot(Nf, r["a"] + r["sigma2"] * Nf + r["q"] * Nf ** 2, color=colour, lw=1.5, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events the pixel has received, N", color=TEXT_2, fontsize=9)
    ax.set_ylabel("Var(I) over pixels (steps²)", color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, title="points: data · lines: a + σ²N + qN², fitted for N ≥ 20",
              title_fontsize=8)
    fig.suptitle("Without forgetting, I's error grows with every event", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "variance_growth.png", dpi=150, facecolor=SURFACE)
    if args.sweep:
        sys.path.insert(0, str(Path(__file__).parent))
        from polarity_check import DIAGNOSTICS, sign_check
        taus = [float(v) for v in args.sweep]
        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
        style(ax)
        ax.axvspan(results["tau_star"]["q25"], results["tau_star"]["q75"], color=GRID, alpha=0.8, lw=0)
        ax.axvline(results["tau_star"]["median"], color=TEXT_2, lw=1)
        ax.text(results["tau_star"]["median"] * 1.08, 0.505, "τ* = 1 / (σ² rate)", color=TEXT_2, fontsize=8)
        ax.axvline(0.1, color=TEXT_2, lw=1, ls="--")
        ax.text(0.108, 0.53, "inherited 0.1 s", color=TEXT_2, fontsize=8)
        results["sweep"] = {}
        for colour, mode in zip(SERIES, ["equal", "pixel"]):
            vals = []
            for tau, raw in zip(taus, args.sweep):
                d = np.fromfile(Path("output") / f"tau_{mode}_{raw}" / "event_diagnostics.bin", dtype=DIAGNOSTICS)
                agree, ok = sign_check(d, d["MRgyro"].astype(np.float64))
                vals.append(float(agree.sum() / ok.sum()))
            results["sweep"][mode] = dict(tau=taus, agreement=vals)
            print(f"sweep {mode:6s}: " + ", ".join(f"tau {t:g}: {v:.3f}" for t, v in zip(taus, vals)))
            ax.plot(taus, vals, "-o", color=colour, lw=2, ms=5, label=f"{mode} steps")
        ax.set_xscale("log")
        ax.set_ylim(0.5, 0.9)
        ax.set_xlabel("decay time τ of I (s)", color=TEXT_2, fontsize=9)
        ax.set_ylabel("polarity = sign(−G · u_gyro)", color=TEXT_2, fontsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")
        fig.suptitle("How long should I remember?", color=TEXT, fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(args.out / "decay_sweep.png", dpi=150, facecolor=SURFACE)
    (args.out / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
