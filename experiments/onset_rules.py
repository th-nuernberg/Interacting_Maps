"""Compare onset rules (events -> O): latest event, hard burst gap, and soft onset by synaptic depression.

Every rule feeds the same O -> N relation (event_maps.normal_flow) and is scored twice:
  (a) normal flow at events near DAVIS frames against the true normal flow (gyro + frame gradient, v_units.py)
  (b) camera angular velocity from the standing constraints of all events against the gyro (r_prototype.py,
      steady state, kappa = inf, J with lens distortion, measurement lifetime 10 ms)

Usage (repository root):  PYTHONPATH=experiments python/.venv/bin/python experiments/onset_rules.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from event_maps import Depression, EventIndex, HardGap, LatestEvent
from r_prototype import EVAL_RATE, basis_flows, gyro_metrics, normal_flow_for_rule, steady_state
from v_units import ground_truth_events, load_calibration, time_surface_report

RULES = [LatestEvent(), HardGap(0.01)] + [Depression(tau) for tau in (0.002, 0.005, 0.01, 0.02, 0.04)] \
        + [Depression(0.01, use=0.5)]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical slots 1-3 (validated all-pairs)


def check_depression_limits():
    """Depression must reduce to the latest event for tau -> 0 and to a single onset for tau -> inf."""
    t = np.array([0.0, 0.001, 0.002, 0.050, 0.051])
    gap = np.array([np.inf, 0.001, 0.001, 0.048, 0.001])
    from event_maps import _depression
    fast, _ = _depression(t, gap, 1e-9, 1.0)
    slow, _ = _depression(t, gap, 1e9, 1.0)
    assert np.allclose(fast, t), fast
    assert np.allclose(slow, 0.0), slow


def plot(reports, names, path):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(1.0, color=TEXT_2, lw=1, zorder=1)
    for colour, name, offset in zip(SERIES, names, [8, -20, 8]):
        bins = reports[name]["ratio_by_normal_speed"]
        mids = [np.sqrt(b["lo"] * b["hi"]) for b in bins]
        ratio = [b["median_ratio"] for b in bins]
        ax.plot(mids, ratio, color=colour, lw=2, label=name, zorder=2)
        ax.scatter(mids, ratio, s=36, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
        # direct label mid-curve, above or below the line so neighbouring curves don't collide
        ax.annotate(name, (mids[4], ratio[4]), xytext=(4, offset), textcoords="offset points", color=TEXT_2,
                    fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4, 6])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("true normal speed |u_n| (px/s)", color=TEXT_2, fontsize=9)
    ax.set_ylabel("estimated / true speed", color=TEXT_2, fontsize=9)
    ax.grid(True, color=GRID, lw=1)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle("Normal speed from neighbour timing, by onset rule (1 = correct)", color=TEXT, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("res/shapes_rotation"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/onset_rules"))
    ap.add_argument("--gt-cache", type=Path, default=Path("experiments/results/v_units"))
    ap.add_argument("--flow-cache", type=Path, default=Path("experiments/results/r_prototype"))
    ap.add_argument("--imu-lag", type=float, default=0.004)
    ap.add_argument("--lifetime", type=float, default=0.01, help="measurement lifetime tau (s)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    check_depression_limits()

    gt = ground_truth_events(args.data, args.gt_cache)
    index = EventIndex.build(gt.t, gt.x, gt.y, gt.p)
    imu = np.loadtxt(args.data / "imu.txt")
    J = basis_flows(load_calibration(args.data / "calib.txt"))
    pix = gt.y.astype(np.int64) * 240 + gt.x
    t_eval = np.arange(1.0, gt.t[-1], 1.0 / EVAL_RATE)

    reports, rows = {}, []
    for rule in RULES:
        a = time_surface_report(index, rule, gt)
        nx, ny, s, var = normal_flow_for_rule(index, rule, args.flow_cache)
        omega, *_ = steady_state(gt.t, pix, nx, ny, s, var, J, np.inf, args.lifetime, t_eval, 5000.0)
        b = gyro_metrics(t_eval, omega, imu, args.imu_lag)
        reports[rule.name] = dict(normal_flow=a, gyro=b, ratio_by_normal_speed=a["ratio_by_normal_speed"])
        rows.append((rule.name, a, b))
        print(f"{rule.name:24s} | N: median ratio {a['median_ratio']:.2f}, IQR(log) {a['log_ratio_iqr']:.2f}, "
              f"dir cos {a['direction_cos_median']:.2f}, by |u_n| "
              + " ".join(f"{x['median_ratio']:.2f}" for x in a["ratio_by_normal_speed"])
              + f" | omega: cos {b['cos_median']:.3f} (mean {b['cos_mean']:.3f}), ratio {b['speed_ratio_median']:.3f}"
              f" [{b['speed_ratio_iqr'][0]:.2f}, {b['speed_ratio_iqr'][1]:.2f}], axis corr {np.round(b['axis_corr'], 3)}",
              flush=True)

    best = max((r for r in RULES if isinstance(r, Depression)), key=lambda r: reports[r.name]["gyro"]["cos_mean"])
    plot(reports, ["latest event", "hard gap 10 ms", best.name], args.out / "speed_ratio_by_rule.png")
    (args.out / "results.json").write_text(json.dumps(
        dict(lifetime=args.lifetime, imu_lag=args.imu_lag, rules={k: dict(normal_flow=v["normal_flow"], gyro=v["gyro"])
                                                                for k, v in reports.items()}), indent=2))


if __name__ == "__main__":
    main()
