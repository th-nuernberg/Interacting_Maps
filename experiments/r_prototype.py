"""Offline prototype: rotation from event timing, as a loop and as its steady state.

Measurements. Every event gives a normal flow from the burst-onset time surface of its 5 x 5 neighbourhood
(experiments/v_units.py): speed s (px/s) along the direction n, with a variance from the plane-fit residuals.
Each pixel keeps its latest measurement as a standing constraint whose weight fades with its age:
    w_p(t) = exp(-(t - t_p) / tau) / sigma_p^2

Energy (one term per piece of evidence):
    E(F, omega) = sum_measured w_p (n_p . F_p - s_p)^2 + kappa sum_all |F_p - J_p omega|^2
J_p maps the camera's angular velocity omega (rad/s) to the pixel's flow (px/s); R = -omega.

Steady state ("direct"): unmeasured pixels follow omega exactly and drop out, leaving
    omega* = argmin sum_measured w'_p (n_p . J_p omega - s_p)^2,   w' = w kappa / (w + kappa)
Loop (the network's structure: F <- omega at every pixel, then omega <- least squares over all pixels' F):
    omega <- omega + A_all^-1 (b - H omega),   A_all = sum_all J^T J,  H = sum a a^T w / (w + kappa),
    b = sum a s w / (w + kappa),  a = J^T n
Its fixed point is omega*; T trips per 1/60 s window are computed in closed form, warm-started.

Usage (repository root):  python/.venv/bin/python experiments/r_prototype.py
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from event_maps import EventIndex, HardGap, normal_flow_all
from v_units import HEIGHT, WIDTH, load_calibration, load_events, rotational_flow

EVAL_RATE = 60.0                   # Hz
TRIPS = [1, 3, 10, 30, 100, 300]   # loop trips per evaluation window
TAUS = [0.01, 0.03, 0.1]           # s, lifetime of a standing measurement


# ----------------------------------------------------------------------------------------------------------------
# Measurements: events -> O (onset rule) -> N (normal flow), cached per rule

def normal_flow_for_rule(index, rule, out, horizon=0.03):
    cache = out / f"normal_flow_{rule.name.replace(' ', '_')}_{horizon * 1000:g}ms.npz"
    if cache.exists():
        d = np.load(cache)
        return d["nx"], d["ny"], d["s"], d["var"]
    stamps, _ = rule(index)
    nx, ny, s, var = normal_flow_all(index, stamps, horizon=horizon)
    np.savez(cache, nx=nx, ny=ny, s=s, var=var)
    return nx, ny, s, var


# ----------------------------------------------------------------------------------------------------------------

def basis_flows(calib, pinhole=False):
    """J for every pixel: flow (px/s) for unit angular velocity about x, y, z. Shape (H*W, 2, 3).
    pinhole=True ignores lens distortion, as the C++ model does."""
    if pinhole:
        calib = dict(calib, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    u, v = xx.ravel().astype(float), yy.ravel().astype(float)
    J = np.zeros((len(u), 2, 3))
    for k in range(3):
        w = np.zeros((len(u), 3)); w[:, k] = 1.0
        fx, fy = rotational_flow(u, v, w, calib)
        J[:, 0, k], J[:, 1, k] = fx, fy
    return J


def estimate(t, pix, nx, ny, s, var, J, kappa, tau, t_eval, max_speed):
    """Per evaluation time: H, b of the standing constraints (latest measurement per pixel, faded)."""
    valid = np.isfinite(s) & (s > 0) & (s < max_speed)
    ev = np.flatnonzero(valid)
    a_all = np.einsum("pij,pi->pj", J[pix[ev]], np.stack([nx[ev], ny[ev]], -1))    # a = J^T n, per event
    last_t = np.full(HEIGHT * WIDTH, -np.inf)
    last_a = np.zeros((HEIGHT * WIDTH, 3)); last_s = np.zeros(HEIGHT * WIDTH); last_w0 = np.zeros(HEIGHT * WIDTH)
    Hs, bs, counts = [], [], []
    start = 0
    for te in t_eval:
        stop = np.searchsorted(t[ev], te, side="right")
        if stop > start:
            sl = slice(start, stop)
            # latest measurement per pixel within this window wins
            pw = pix[ev[sl]]
            rev = np.unique(pw[::-1], return_index=True)[1]
            keep = (stop - start - 1) - rev
            q = pw[keep]
            last_t[q] = t[ev[sl]][keep]; last_a[q] = a_all[sl][keep]
            last_s[q] = s[ev[sl]][keep]; last_w0[q] = 1.0 / var[ev[sl]][keep]
            start = stop
        age = te - last_t
        active = age < 8 * tau
        w = last_w0[active] * np.exp(-age[active] / tau)
        # loop step weight; the steady-state weight is kappa * we (just w for kappa = inf, F following omega fully)
        we = w if np.isinf(kappa) else w / (w + kappa)
        a = last_a[active]
        Hs.append(np.einsum("p,pi,pj->ij", we, a, a))
        bs.append(np.einsum("p,pi,p->i", we, a, last_s[active]))
        counts.append(int(active.sum()))
    return np.array(Hs), np.array(bs), np.array(counts)


def steady_state(t, pix, nx, ny, s, var, J, kappa, tau, t_eval, max_speed):
    """omega at each evaluation time from the standing constraints (the loop's fixed point)."""
    H, b, counts = estimate(t, pix, nx, ny, s, var, J, kappa, tau, t_eval, max_speed)
    return np.array([np.linalg.lstsq(h, v, rcond=None)[0] for h, v in zip(H, b)]), H, b, counts


def gyro_metrics(t_eval, est, imu, lag, min_speed=1.0):
    g = np.stack([np.interp(t_eval + lag, imu[:, 0], imu[:, 4 + k]) for k in range(3)], -1)
    ok = np.all(np.isfinite(est), 1) & (np.linalg.norm(g, axis=1) > min_speed) & (t_eval > 2.0)
    e, g = est[ok], g[ok]
    cos = np.sum(e * g, 1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(g, axis=1))
    ratio = np.linalg.norm(e, axis=1) / np.linalg.norm(g, axis=1)
    return dict(cos_median=float(np.median(cos)), cos_mean=float(np.mean(cos)),
                speed_ratio_median=float(np.median(ratio)),
                speed_ratio_iqr=[float(v) for v in np.percentile(ratio, [25, 75])],
                axis_corr=[float(np.corrcoef(e[:, k], g[:, k])[0, 1]) for k in range(3)],
                windows=int(ok.sum()))


# ----------------------------------------------------------------------------------------------------------------
# Plot: one panel per axis, gyro vs steady-state estimate (two series, legend + direct labels)

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"


def plot(t_eval, est, imu, lag, path, window=(20.0, 30.0)):
    g = np.stack([np.interp(t_eval + lag, imu[:, 0], imu[:, 4 + k]) for k in range(3)], -1)
    sel = (t_eval >= window[0]) & (t_eval <= window[1])
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, facecolor=SURFACE)
    for k, ax in enumerate(axes):
        ax.set_facecolor(SURFACE)
        ax.plot(t_eval[sel], g[sel, k], color=SERIES_1, lw=2, label="gyro")
        ax.plot(t_eval[sel], est[sel, k], color=SERIES_2, lw=2, label="from event timing (steady state)")
        ax.set_ylabel(f"ω_{'xyz'[k]} (rad/s)", color=TEXT_2, fontsize=9)
        ax.grid(True, color=GRID, lw=1)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=False, fontsize=9, labelcolor=TEXT)
    axes[-1].set_xlabel("time (s)", color=TEXT_2, fontsize=9)
    fig.suptitle("Camera angular velocity: gyro vs estimate from event timing alone (τ = 10 ms)", color=TEXT,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("res/shapes_rotation"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/r_prototype"))
    ap.add_argument("--cache", type=Path, default=Path("experiments/results/v_units/events_cache.npz"))
    ap.add_argument("--max-speed", type=float, default=5000.0, help="discard normal speeds above this (px/s)")
    ap.add_argument("--imu-lag", type=float, default=0.004, help="s, from experiments/v_units.py")
    ap.add_argument("--step0", action="store_true",
                    help="only compare the steady state for kappa in {median precision, inf} x J {distorted, pinhole}")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t, x, y, p = load_events(args.data / "events.txt", args.cache)
    calib = load_calibration(args.data / "calib.txt")
    imu = np.loadtxt(args.data / "imu.txt")
    pix = y.astype(np.int64) * WIDTH + x

    index = EventIndex.build(t, x, y, p)
    nx, ny, s, var = normal_flow_for_rule(index, HardGap(0.01), args.out)
    ok = np.isfinite(s) & (s < args.max_speed)
    sigma_s = np.sqrt(var[ok])
    kappa = 1.0 / np.median(sigma_s) ** 2       # model precision = typical measurement precision (open)
    print(f"normal flow for {ok.mean():.0%} of events; median s {np.median(s[ok]):.0f} px/s, "
          f"median sigma_s {np.median(sigma_s):.0f} px/s -> kappa = 1 / ({np.median(sigma_s):.0f} px/s)^2")

    if args.step0:
        t_eval = np.arange(1.0, t[-1], 1.0 / EVAL_RATE)
        tau, rows = 0.01, {}
        for j_label, pinhole in [("distorted", False), ("pinhole", True)]:
            Jv = basis_flows(calib, pinhole=pinhole)
            for k_label, k in [("median precision", kappa), ("inf", np.inf)]:
                H, b, _ = estimate(t, pix, nx, ny, s, var, Jv, k, tau, t_eval, args.max_speed)
                direct = np.array([np.linalg.lstsq(h, v, rcond=None)[0] for h, v in zip(H, b)])
                d = gyro_metrics(t_eval, direct, imu, args.imu_lag)
                rows[f"J {j_label}, kappa {k_label}"] = d
                print(f"J {j_label:9s} kappa {k_label:16s}: cos median {d['cos_median']:+.3f} mean {d['cos_mean']:+.3f}, "
                      f"|omega| ratio {d['speed_ratio_median']:.3f} [{d['speed_ratio_iqr'][0]:.2f}, "
                      f"{d['speed_ratio_iqr'][1]:.2f}], axis corr {np.round(d['axis_corr'], 3)}")
        (args.out / "step0.json").write_text(json.dumps(dict(tau=tau, kappa=float(kappa), results=rows), indent=2))
        return

    J = basis_flows(calib)
    A_all = np.einsum("pki,pkj->ij", J, J)
    A_inv = np.linalg.inv(A_all)
    t_eval = np.arange(1.0, t[-1], 1.0 / EVAL_RATE)

    results = dict(kappa=float(kappa), sigma_s_median=float(np.median(sigma_s)), imu_lag=args.imu_lag, taus={})
    for tau in TAUS:
        H, b, counts = estimate(t, pix, nx, ny, s, var, J, kappa, tau, t_eval, args.max_speed)
        direct = np.array([np.linalg.lstsq(h, v, rcond=None)[0] for h, v in zip(H, b)])
        # loop: omega <- omega + A_all^-1 (b - H omega), T trips per window, warm-started
        lam = np.array([np.linalg.eigvals(A_inv @ h).real.min() for h in H])
        entry = dict(direct=gyro_metrics(t_eval, direct, imu, args.imu_lag),
                     measured_pixels_median=int(np.median(counts)),
                     trips_for_1_over_e_median=float(np.median(1.0 / np.maximum(lam, 1e-12))),
                     loop={})
        for T in TRIPS:
            omega = np.zeros(3); est = np.zeros_like(direct)
            for i, (h, v) in enumerate(zip(H, b)):
                step = np.eye(3) - A_inv @ h
                omega = direct[i] + np.linalg.matrix_power(step, T) @ (omega - direct[i])
                est[i] = omega
            gap = np.linalg.norm(est - direct, axis=1) / np.maximum(np.linalg.norm(direct, axis=1), 1e-9)
            entry["loop"][T] = dict(gyro=gyro_metrics(t_eval, est, imu, args.imu_lag),
                                    relative_gap_to_steady_state_median=float(np.median(gap[t_eval > 2.0])))
        results["taus"][tau] = entry
        if tau == 0.01:
            plot(t_eval, direct, imu, args.imu_lag, args.out / "omega_vs_gyro.png")
        d = entry["direct"]
        print(f"\ntau = {tau * 1000:g} ms: {entry['measured_pixels_median']} measured pixels per window (median); "
              f"loop needs ~{entry['trips_for_1_over_e_median']:.0f} trips per 1/e")
        print(f"  steady state   : cos {d['cos_median']:+.2f} (mean {d['cos_mean']:+.2f}), |omega| ratio "
              f"{d['speed_ratio_median']:.2f} [{d['speed_ratio_iqr'][0]:.2f}, {d['speed_ratio_iqr'][1]:.2f}], "
              f"axis corr {np.round(d['axis_corr'], 2)}")
        for T in TRIPS:
            L = entry["loop"][T]
            print(f"  loop {T:4d} trips: cos {L['gyro']['cos_median']:+.2f}, |omega| ratio "
                  f"{L['gyro']['speed_ratio_median']:.2f}, gap to steady state {L['relative_gap_to_steady_state_median']:.2f}")
    (args.out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
