"""Separate ON/OFF contrast steps instead of a leak? (shapes_rotation)

OFF events outnumber ON events 1.36 : 1 per pixel while the scene does not get darker, so c_ON ~ 1.36 c_OFF
(docs/assumptions.md, W3). With equal steps I drifts and the decay (tau = 0.1 s) is needed. This script compares
C++ runs with --contrastSteps equal | global | pixel, with and without the decay (--dumpEvents, see
polarity_check.py), on:

  1. drift: I at event pixels over time (median and 5-95 %);
  2. polarity agreement sign(-G . u_gyro) = p, for ON and OFF events separately, and |G| at ON vs OFF events;
  3. G against the frame gradient of log I (optional, slow);
  4. the global ratio estimate OFF/ON over time (from the event file).

Usage (repository root):
    python/.venv/bin/python experiments/contrast_steps.py "equal, decay=output/units_default" \\
        "equal=output/contrast_equal_nodecay" "global=output/contrast_global_nodecay" ...
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
from polarity_check import DIAGNOSTICS, SURFACE, TEXT, TEXT_2, GRID, style, sign_check, frame_reference  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1a21b"]   # categorical slots 1-5, fixed order


def analyse(d):
    u = d["MRgyro"].astype(np.float64)
    agree, ok = sign_check(d, u)
    on, off = d["p"] > 0, d["p"] < 0
    g = np.linalg.norm(d["G"].astype(np.float64), axis=1)
    edges = np.arange(0, 60.01, 1.0)
    idx = np.digitize(d["t"], edges) - 1
    med, lo, hi = [], [], []
    for b in range(len(edges) - 1):
        v = d["I"][idx == b]
        med.append(float(np.median(v)))
        lo.append(float(np.percentile(v, 5)))
        hi.append(float(np.percentile(v, 95)))
    late = d["t"] > 50
    return dict(agreement=float(agree.sum() / ok.sum()),
                agreement_on=float(agree[on].sum() / ok[on].sum()), agreement_off=float(agree[off].sum() / ok[off].sum()),
                G_on=float(np.median(g[on])), G_off=float(np.median(g[off])),
                I_median_late=float(np.median(d["I"][late])), I_abs99_late=float(np.percentile(np.abs(d["I"][late]), 99)),
                t=(0.5 * (edges[1:] + edges[:-1])).tolist(), I_med=med, I_lo=lo, I_hi=hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=output/<run> with event_diagnostics.bin")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--data", type=Path, default=Path("res/shapes_rotation"))
    ap.add_argument("--cache", type=Path, default=Path("experiments/results/v_units"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/contrast_steps"))
    ap.add_argument("--frames", action="store_true", help="also compare G with the frame gradient (slow)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [r.split("=", 1) for r in args.runs]
    names = [n for n, _ in runs]
    colours = {n: SERIES[i] for i, n in enumerate(names)}
    gt = None
    if args.frames:
        import v_units
        gt = v_units.ground_truth_events(args.data, args.cache)
    results = {}
    for name, path in runs:
        d = np.fromfile(Path(path) / "event_diagnostics.bin", dtype=DIAGNOSTICS)
        r = analyse(d)
        if gt is not None:
            f = frame_reference(d, args.stride, gt)
            r["cos_G_frame_median"] = f["cos_G_frame_median"]
            r["cos_G_frame_positive"] = f["cos_G_frame_positive"]
        results[name] = r
        print(f"{name:22s} agreement {r['agreement']:.3f} (ON {r['agreement_on']:.3f}, OFF {r['agreement_off']:.3f}); "
              f"|G| ON {r['G_on']:.2f} OFF {r['G_off']:.2f}; I after 50 s: median {r['I_median_late']:+.1f}, "
              f"99 % |I| {r['I_abs99_late']:.1f}" + (f"; cos(G, frame) {r['cos_G_frame_median']:.3f}" if gt is not None else ""))

    # Global ratio estimate over time, from the events
    c = np.load(args.cache / "events_cache.npz")
    t, p = c["t"], c["p"]
    marks = np.searchsorted(t, np.arange(0.01, 60.0, 0.01))
    n_on = np.cumsum(p > 0)[np.minimum(marks, len(t) - 1)]
    n_off = np.cumsum(p < 0)[np.minimum(marks, len(t) - 1)]
    ratio_t, ratio = np.arange(0.01, 60.0, 0.01), (n_off + 1) / (n_on + 1)
    print(f"global OFF/ON ratio: after 0.1 s {ratio[9]:.3f}, 1 s {ratio[99]:.3f}, 10 s {ratio[999]:.3f}, 60 s {ratio[-1]:.3f}")

    # Figure 1: I at event pixels over time (median and 5-95 %), one panel per run, shared y
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharey=True, facecolor=SURFACE)
    for ax, n in zip(np.atleast_1d(axes), names):
        style(ax)
        r = results[n]
        ax.fill_between(r["t"], r["I_lo"], r["I_hi"], color=colours[n], alpha=0.25, lw=0)
        ax.plot(r["t"], r["I_med"], color=colours[n], lw=2)
        ax.axhline(0, color=TEXT_2, lw=1)
        ax.set_title(n, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("I at event pixels (event steps)\nmedian, 5–95 %", color=TEXT_2, fontsize=9)
    fig.suptitle("Does I stay bounded?", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "drift_of_I.png", dpi=150, facecolor=SURFACE)

    # Figure 2: polarity agreement for ON and OFF events (grouped horizontal bars)
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.75 * len(names)), facecolor=SURFACE)
    style(ax)
    for k, (key, label, shade) in enumerate([("agreement_on", "ON events", 1.0), ("agreement_off", "OFF events", 0.45)]):
        ypos = np.arange(len(names)) + (k - 0.5) * 0.4
        vals = [results[n][key] for n in names]
        ax.barh(ypos, vals, height=0.36, color=[colours[n] for n in names], alpha=shade)
        for yv, v in zip(ypos, vals):
            ax.text(v + 0.004, yv, f"{v:.3f} {label}", va="center", fontsize=8, color=TEXT)
    ax.axvline(0.5, color=TEXT_2, lw=1, ls="--")
    ax.set_yticks(range(len(names)), names, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("fraction of events with polarity = sign(−G · u_gyro)", color=TEXT_2, fontsize=9)
    fig.suptitle("Polarity agreement for ON and OFF events", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "agreement_on_off.png", dpi=150, facecolor=SURFACE)

    # Figure 3: global ratio estimate over time
    fig, ax = plt.subplots(figsize=(7, 3), facecolor=SURFACE)
    style(ax)
    ax.plot(ratio_t, ratio, color=SERIES[0], lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("OFF / ON events so far", color=TEXT_2, fontsize=9)
    fig.suptitle("Estimate of c_ON / c_OFF from the event counts", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "ratio_estimate.png", dpi=150, facecolor=SURFACE)

    (args.out / "summary.json").write_text(json.dumps(
        {n: {k: v for k, v in r.items() if k not in ("t", "I_med", "I_lo", "I_hi")} for n, r in results.items()}
        | {"global_ratio_60s": float(ratio[-1])}, indent=2))


if __name__ == "__main__":
    main()
