"""Is the network's G usable? Polarity and direction checks against event timing and the frames (shapes_rotation).

Brightness constancy predicts the sign of every event: polarity = sign(V) = sign(-G . u), with u the image motion
at the pixel. The C++ run with --dumpEvents writes, for every n-th event, the maps just before the event's own
I/G update (src/main.cpp, EventDiagnostics). This script scores:

  1. sign agreement of the polarity with -G . u for four motions u: the gyro (reference), M R (network),
     the stored F, and the event's own normal flow N;
  2. the same, by time since the pixel's previous event and by |G|;
  3. the direction of G against N (both are normals of the edge) and against the frame gradient of log I
     (ground truth from experiments/v_units.py, joined by event index);
  4. how often I sits at its clamp bounds (0, 255) around the event.

Usage (repository root):
    ./InteractingMaps --startTime 0 --endTime 60 --resultsDirectory polarity_default \\
        --writeFrames false --dumpEvents 20            # and e.g. --decay false into polarity_nodecay
    python/.venv/bin/python experiments/polarity_check.py default=output/polarity_default \\
        nodecay=output/polarity_nodecay --stride 20 [--i-bounds 0 255 for runs before I was unbounded]
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

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d1a21b"]   # categorical slots 1-5, fixed order

DIAGNOSTICS = np.dtype([("t", "f8"), ("y", "f4"), ("x", "f4"), ("p", "f4"), ("G", "f4", 2), ("F", "f4", 2),
                        ("MR", "f4", 2), ("MRgyro", "f4", 2), ("Nvalid", "f4"), ("N", "f4", 2), ("Nspeed", "f4"),
                        ("I", "f4"), ("Imin", "f4"), ("Imax", "f4"), ("dtPrevious", "f4")], align=True)
MOTIONS = {"gyro flow (reference)": "MRgyro", "M R (network rotation)": "MR", "stored F": "F",
           "N (this event's normal flow)": "N"}
DT_EDGES = np.array([0, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.3, 1.0, np.inf])


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)


def load(run: Path):
    d = np.fromfile(run / "event_diagnostics.bin", dtype=DIAGNOSTICS)
    motion = {name: d[field].astype(np.float64) for name, field in MOTIONS.items()}
    motion["N (this event's normal flow)"] = d["N"] * d["Nspeed"][:, None]
    return d, motion


def sign_check(d, u, valid=None):
    """Fraction of events whose polarity equals sign(-G . u), over events where G . u is not 0."""
    v_pred = -np.sum(d["G"].astype(np.float64) * u, axis=1)
    ok = np.abs(v_pred) > 1e-6
    if valid is not None:
        ok &= valid
    return (np.sign(v_pred) == d["p"]) & ok, ok


def binned(values, agree, ok, edges):
    idx = np.digitize(values, edges) - 1
    rate, count = [], []
    for b in range(len(edges) - 1):
        sel = ok & (idx == b)
        count.append(int(sel.sum()))
        rate.append(float(agree[sel].sum() / sel.sum()) if sel.sum() >= 200 else np.nan)
    return np.array(rate), np.array(count)


def analyse(name, run, bounds=None):
    d, motion = load(run)
    g = d["G"].astype(np.float64)
    g_norm = np.linalg.norm(g, axis=1)
    out = dict(run=str(run), events=len(d), G_zero=float(np.mean(g_norm < 1e-6)),
               N_valid=float(np.mean(d["Nvalid"] > 0)))
    lo, hi = bounds if bounds else (-np.inf, np.inf)      # I's clamp bounds of the run (0 255 before 2026-11-06)
    out.update(I_at_bound=float(np.mean((d["Imin"] <= lo) | (d["Imax"] >= hi))),
               I_at_upper=float(np.mean(d["Imax"] >= hi)), I_at_lower=float(np.mean(d["Imin"] <= lo)), bounds=[lo, hi])
    out["sign"] = {}
    for m, u in motion.items():
        valid = d["Nvalid"] > 0 if m.startswith("N ") else None
        agree, ok = sign_check(d, u, valid)
        out["sign"][m] = dict(agreement=float(agree.sum() / ok.sum()), scored=float(ok.mean()))
    # By time since the pixel's previous event and by |G|, against the reference motion
    agree, ok = sign_check(d, motion["gyro flow (reference)"])
    out["by_dt"], out["by_dt_count"] = [x.tolist() for x in binned(d["dtPrevious"], agree, ok, DT_EDGES)]
    g_edges = np.quantile(g_norm[ok], np.linspace(0, 1, 11))
    out["G_deciles"] = g_edges.tolist()
    out["by_G"] = binned(g_norm, agree, ok, g_edges)[0].tolist()
    out["G_median_by_dt"] = [float(np.median(g_norm[(np.digitize(d["dtPrevious"], DT_EDGES) - 1) == b]))
                             if np.any((np.digitize(d["dtPrevious"], DT_EDGES) - 1) == b) else np.nan
                             for b in range(len(DT_EDGES) - 1)]
    # Direction: G should be parallel to the edge normal N (up to sign)
    nv = (d["Nvalid"] > 0) & (g_norm > 1e-6)
    n = d["N"][nv].astype(np.float64)
    cos_gn = np.sum(g[nv] * n, axis=1) / g_norm[nv]
    out["abs_cos_G_N_median"] = float(np.median(np.abs(cos_gn)))
    out["abs_cos_G_N"] = np.abs(cos_gn)
    # I over time (saturation)
    out["t"], out["Imin"], out["Imax"] = d["t"], d["Imin"], d["Imax"]
    print(f"{name}: {len(d)} events, G = 0 at {out['G_zero']:.1%}, N valid {out['N_valid']:.1%}, "
          f"I stencil at a bound {out['I_at_bound']:.1%}")
    for m, s in out["sign"].items():
        print(f"   sign agreement with -G.u, u = {m:30s} {s['agreement']:.3f}  (scored {s['scored']:.1%})")
    print(f"   |cos(G, N)| median {out['abs_cos_G_N_median']:.3f}")
    return d, out


def frame_reference(d, stride, gt):
    """Join with the v_units ground truth (events near DAVIS frames): network G vs frame gradient of log I."""
    # Match by (time, x, y), not by index: the C++ times are floats, and events on 1 s slice boundaries are
    # processed twice (slice_events includes both ends), which shifts the index by a few dozen events.
    te, xe, ye = gt.t[gt.events], gt.x[gt.events], gt.y[gt.events]
    pos = np.searchsorted(d["t"], te)
    rows = np.full(len(te), -1)
    for off in range(-4, 5):
        r = np.clip(pos + off, 0, len(d) - 1)
        hit = (rows < 0) & (np.abs(d["t"][r] - te) < 2e-5) & (d["x"][r] == xe) & (d["y"][r] == ye)
        rows[hit] = r[hit]
    pick = np.flatnonzero(rows >= 0)
    rows = rows[pick]
    print(f"joined {len(pick)} of {len(te)} ground-truth events with the dump (stride {stride}: expect ~{len(te) // stride})")
    g_true = np.stack([gt.gy[pick], gt.gx[pick]], 1)              # (y, x), log units per px
    u_true = np.stack([gt.uy[pick], gt.ux[pick]], 1)
    g_net = d["G"][rows].astype(np.float64)
    p = d["p"][rows]
    nz = np.linalg.norm(g_net, axis=1) > 1e-6
    cos = np.sum(g_net * g_true, 1) / (np.linalg.norm(g_net, axis=1) * np.linalg.norm(g_true, axis=1) + 1e-12)
    res = dict(joined=int(len(pick)), G_zero=float(np.mean(~nz)), cos_G_frame=cos[nz],
               cos_G_frame_median=float(np.median(cos[nz])), cos_G_frame_positive=float(np.mean(cos[nz] > 0)),
               frame_sign_agreement=float(np.mean(np.sign(-np.sum(g_true * u_true, 1)) == p)),
               net_sign_agreement_same_events=float(np.mean(np.sign(-np.sum(g_net[nz] * u_true[nz], 1)) == p[nz])))
    print(f"   frames: cos(G_net, grad log I) median {res['cos_G_frame_median']:.3f}, positive {res['cos_G_frame_positive']:.1%}; "
          f"sign agreement frame G {res['frame_sign_agreement']:.3f} vs network G {res['net_sign_agreement_same_events']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=output/<run> with event_diagnostics.bin")
    ap.add_argument("--stride", type=int, default=20, help="--dumpEvents of the runs")
    ap.add_argument("--data", type=Path, default=Path("res/shapes_rotation"))
    ap.add_argument("--cache", type=Path, default=Path("experiments/results/v_units"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/polarity_check"))
    ap.add_argument("--no-frames", action="store_true", help="skip the frame-gradient join")
    ap.add_argument("--i-bounds", type=float, nargs=2, help="clamp bounds of I in the runs (old runs: 0 255)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [r.split("=", 1) for r in args.runs]
    results, frames = {}, {}
    gt = None
    if not args.no_frames:
        import v_units
        gt = v_units.ground_truth_events(args.data, args.cache)
    for name, path in runs:
        d, results[name] = analyse(name, Path(path), args.i_bounds)
        if gt is not None:
            frames[name] = frame_reference(d, args.stride, gt)

    names = [n for n, _ in runs]
    colours = {n: SERIES[i] for i, n in enumerate(names)}

    # Figure 1: sign agreement per motion source and run (horizontal bars, chance and frame baseline marked)
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    style(ax)
    motions = list(MOTIONS)
    h = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [results[n]["sign"][m]["agreement"] for m in motions]
        ypos = np.arange(len(motions)) + (i - (len(names) - 1) / 2) * h
        ax.barh(ypos, vals, height=h * 0.9, color=colours[n], label=n)
        for yv, v in zip(ypos, vals):
            ax.text(v + 0.005, yv, f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax.axvline(0.5, color=TEXT_2, lw=1, ls="--")
    ax.text(0.505, len(motions) - 0.45, "chance", fontsize=8, color=TEXT_2)
    if frames:
        fb = np.mean([f["frame_sign_agreement"] for f in frames.values()])
        ax.axvline(fb, color=TEXT_2, lw=1)
        ax.text(fb - 0.005, len(motions) - 0.45, f"frame gradient + gyro {fb:.2f}", fontsize=8, color=TEXT_2, ha="right")
    ax.set_yticks(range(len(motions)), motions, color=TEXT, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("fraction of events with polarity = sign(−G · u)", color=TEXT_2, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT, title="network G from run",
              title_fontsize=8)
    fig.suptitle("Does the network's G predict the polarity of the next event?", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "sign_agreement.png", dpi=150, facecolor=SURFACE)

    # Figure 2: agreement and |G| by time since the pixel's previous event (small multiples, shared x)
    centres = np.sqrt(np.maximum(DT_EDGES[:-1], 1e-4) * np.where(np.isinf(DT_EDGES[1:]), 3.0, DT_EDGES[1:]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURFACE)
    for ax in axes:
        style(ax)
    for n in names:
        axes[0].plot(centres, results[n]["by_dt"], "-o", color=colours[n], lw=2, ms=4, label=n)
        axes[1].plot(centres, results[n]["G_median_by_dt"], "-o", color=colours[n], lw=2, ms=4, label=n)
        share = np.array(results[n]["by_dt_count"]) / np.sum(results[n]["by_dt_count"])
        axes[2].plot(centres, share, "-o", color=colours[n], lw=2, ms=4, label=n)
    for ax, title in zip(axes, ["sign agreement (gyro flow)", "median |G| (I units / px)", "share of events"]):
        ax.set_xscale("log")
        ax.set_xlabel("time since the pixel's previous event (s)", color=TEXT_2, fontsize=9)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
    axes[0].axhline(0.5, color=TEXT_2, lw=1, ls="--")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(args.out / "by_time_since_previous_event.png", dpi=150, facecolor=SURFACE)

    # Figure 3: direction of G (vs N, vs the frame gradient)
    fig, axes = plt.subplots(1, 2 if frames else 1, figsize=(11 if frames else 6, 3.6), facecolor=SURFACE,
                             squeeze=False)
    axes = axes[0]
    bins = np.linspace(0, 1, 41)
    for n in names:
        style(axes[0])
        axes[0].hist(results[n]["abs_cos_G_N"], bins=bins, density=True, histtype="step", lw=2, color=colours[n],
                     label=n)
    ref = 2 / np.pi / np.sqrt(1 - np.clip(bins[:-1] + 0.0125, 0, 0.999) ** 2)
    axes[0].plot(bins[:-1] + 0.0125, ref, color=TEXT_2, lw=1, ls="--", label="random directions")
    axes[0].set_xlabel("|cos| between G and the normal-flow direction N", color=TEXT_2, fontsize=9)
    axes[0].set_title("Is G parallel to the edge normal from timing?", color=TEXT, fontsize=10, loc="left")
    axes[0].set_ylim(0, 5)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT)
    if frames:
        style(axes[1])
        b2 = np.linspace(-1, 1, 41)
        for n in names:
            axes[1].hist(frames[n]["cos_G_frame"], bins=b2, density=True, histtype="step", lw=2, color=colours[n],
                         label=n)
        axes[1].axhline(0.5, color=TEXT_2, lw=1, ls="--", label="random directions")
        axes[1].set_xlabel("cos between network G and the frame gradient of log I", color=TEXT_2, fontsize=9)
        axes[1].set_title("Does G point the right way? (events near frames)", color=TEXT, fontsize=10, loc="left")
        axes[1].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out / "direction_of_G.png", dpi=150, facecolor=SURFACE)

    # Figure 4: I around the events over time (share of events whose 3x3 stencil touches a clamp bound)
    fig, ax = plt.subplots(figsize=(9, 3.2), facecolor=SURFACE)
    style(ax)
    edges = np.arange(0, 60.01, 1.0)
    for n in names:
        r = results[n]
        at = ((r["Imin"] <= r["bounds"][0]) | (r["Imax"] >= r["bounds"][1])).astype(float)
        share = np.histogram(r["t"], edges, weights=at)[0] / np.maximum(np.histogram(r["t"], edges)[0], 1)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), share, color=colours[n], lw=2, label=n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("share of events", color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("How often is I clamped (0 or 255) next to an event?", color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.out / "clamping_of_I.png", dpi=150, facecolor=SURFACE)

    summary = {n: {k: v for k, v in r.items() if k not in ("abs_cos_G_N", "t", "Imin", "Imax")}
               for n, r in results.items()}
    for n, f in frames.items():
        summary[n]["frames"] = {k: v for k, v in f.items() if k != "cos_G_frame"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
