"""V-unit experiment, stage A: which reading of events as V = d log I / dt has the right time unit?

Ground truth, independent of the network (rotation-only recording, static scene):
    u       = image motion of the event pixel from the gyro (px/s), lens distortion included
    grad    = spatial gradient of log intensity from the nearest DAVIS frame (log units / px)
    V_true  = -grad . u                                  (brightness constancy, log units / s)

Each candidate estimator maps an event's polarity and timing to V (with contrast threshold c = 1, i.e. in
events / s). Per speed bin we fit V_est = k V_true. The right time unit gives a k that does not depend on the
image speed |u| of the event pixel (speed exponent beta = 0, k ~ |u|^beta); its level is 1 / c. The current constant V gives
beta = -1.

Usage (from the repository root, with the python/.venv):
    python/.venv/bin/python experiments/v_units.py --data res/shapes_rotation --out experiments/results/v_units
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

WIDTH, HEIGHT = 240, 180
SPEED_EDGES = np.array([20, 40, 80, 120, 160, 240, 320, 480, 640, 960, 1280, 1920.0])   # image speed |u| in px/s
LEAKY_TAUS = [0.001, 0.003, 0.01, 0.03]                                         # s
BIN_LENGTH = 0.01                                                               # s, paper-style bins
MIN_EVENTS_PER_BIN = 1000


# ----------------------------------------------------------------------------------------------------------------
# Data

def load_events(path: Path, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["x"], d["y"], d["p"]
    with open(path) as f:
        first = f.readline()
    raw = np.fromfile(path, sep=" ") if first[0].isdigit() else np.array(
        open(path).read().split("\n", 1)[1].split(), dtype=np.float64)
    raw = raw.reshape(-1, 4)
    t, x, y = raw[:, 0], raw[:, 1].astype(np.int16), raw[:, 2].astype(np.int16)
    p = (raw[:, 3] * 2 - 1).astype(np.int8)                    # {0, 1} -> {-1, +1}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, t=t, x=x, y=y, p=p)
    return t, x, y, p


def load_calibration(path: Path):
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = np.loadtxt(path)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)


def frame_gradients(data: Path):
    """Smoothed gradient of log intensity per frame, (log units / px), arrays indexed [y, x]."""
    times, grads = [], []
    for line in open(data / "images.txt"):
        t, name = line.split()
        img = mpimg.imread(data / name).astype(np.float64)
        if img.ndim == 3:
            img = img[..., 0]
        if img.max() <= 1.0:
            img = img * 255.0
        log_i = gaussian_filter(np.log(img + 1.0), sigma=1.0)
        gy, gx = np.gradient(log_i)
        times.append(float(t))
        grads.append(np.stack([gx, gy]).astype(np.float32))
    return np.array(times), np.stack(grads)


# ----------------------------------------------------------------------------------------------------------------
# Camera model (radial-tangential distortion as in calib.txt)

def distort(xn, yn, c):
    r2 = xn * xn + yn * yn
    radial = 1 + c["k1"] * r2 + c["k2"] * r2 * r2 + c["k3"] * r2 * r2 * r2
    xd = xn * radial + 2 * c["p1"] * xn * yn + c["p2"] * (r2 + 2 * xn * xn)
    yd = yn * radial + c["p1"] * (r2 + 2 * yn * yn) + 2 * c["p2"] * xn * yn
    return c["fx"] * xd + c["cx"], c["fy"] * yd + c["cy"]


def undistort(u, v, c, iterations=20):
    xd, yd = (u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"]
    xn, yn = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = xn * xn + yn * yn
        radial = 1 + c["k1"] * r2 + c["k2"] * r2 * r2 + c["k3"] * r2 * r2 * r2
        dx = 2 * c["p1"] * xn * yn + c["p2"] * (r2 + 2 * xn * xn)
        dy = c["p1"] * (r2 + 2 * yn * yn) + 2 * c["p2"] * xn * yn
        xn, yn = (xd - dx) / radial, (yd - dy) / radial
    return xn, yn


def rotational_flow(u, v, omega, c, dt=1e-4):
    """Pixel velocity (px/s) of static points for a camera rotating with omega (rad/s, camera frame x right,
    y down, z forward). Camera-frame points move as dP/dt = -omega x P."""
    xn, yn = undistort(u, v, c)
    P = np.stack([xn, yn, np.ones_like(xn)], -1)
    P2 = P - np.cross(omega, P) * dt
    u2, v2 = distort(P2[:, 0] / P2[:, 2], P2[:, 1] / P2[:, 2], c)
    return (u2 - u) / dt, (v2 - v) / dt


# ----------------------------------------------------------------------------------------------------------------
# Candidate V estimators, evaluated at sampled events. Arrays are sorted by pixel, then time.

def previous_gap(key, t):
    """Time since the previous event with the same key (inf if none); key/t sorted by key then time."""
    gap = np.full(len(t), np.inf)
    same = key[1:] == key[:-1]
    gap[1:][same] = (t[1:] - t[:-1])[same]
    return gap


def leaky_rate(idx, pix, t, p, tau, max_lag=2000):
    """sum_{j <= i, same pixel} p_j exp(-(t_i - t_j) / tau) / tau, for sorted positions idx."""
    total = p[idx] / tau
    active = np.ones(len(idx), bool)
    for lag in range(1, max_lag):
        j = idx - lag
        active &= j >= 0
        jj = np.where(active, j, 0)
        active &= (pix[jj] == pix[idx]) & (t[idx] - t[jj] < 8 * tau)
        if not active.any():
            break
        with np.errstate(over="ignore", invalid="ignore"):   # inactive entries are discarded by the where
            total = total + np.where(active, p[jj] * np.exp(-(t[idx] - t[jj]) / tau) / tau, 0.0)
    return total


def bin_rate(idx, pix, t, p, length):
    """Net events of the pixel in the fixed bin containing the event, per second (the paper's V)."""
    key = pix.astype(np.float64) * 100.0 + t                  # monotone in (pixel, time); t < 100 s
    start = np.floor(t[idx] / length) * length
    lo = np.searchsorted(key, pix[idx] * 100.0 + start, side="left")
    hi = np.searchsorted(key, pix[idx] * 100.0 + start + length, side="left")
    cum = np.concatenate([[0], np.cumsum(p.astype(np.int64))])
    return (cum[hi] - cum[lo]) / length


def sample_gradient(frame_grad, frame_idx, xf, yf):
    """Bilinear sample of the frame gradient at sub-pixel positions (clipped to the image)."""
    xf = np.clip(xf, 0, WIDTH - 1.001)
    yf = np.clip(yf, 0, HEIGHT - 1.001)
    x0, y0 = np.floor(xf).astype(int), np.floor(yf).astype(int)
    ax, ay = xf - x0, yf - y0
    out = []
    for c in range(2):
        g = frame_grad[:, c]
        out.append((1 - ay) * ((1 - ax) * g[frame_idx, y0, x0] + ax * g[frame_idx, y0, x0 + 1])
                   + ay * ((1 - ax) * g[frame_idx, y0 + 1, x0] + ax * g[frame_idx, y0 + 1, x0 + 1]))
    return out


def time_surface_flow(idx, x, y, p, t, order, radius=2, horizon=0.03, burst_gap=None):
    """Normal flow from arrival times: fit t_q - t_e = a dx + b dy to the most recent same-polarity event of each
    neighbour q (within radius, at most horizon before the event). With burst_gap, every time is replaced by the
    onset of its burst (same pixel and polarity, gaps < burst_gap), which marks when the edge arrived.
    Returns (vx, vy) in px/s and the neighbour count."""
    key_pix = (y.astype(np.int64) * WIDTH + x) * 2 + (p > 0)
    o = np.argsort(key_pix, kind="stable")                  # by (pixel, polarity), then time
    key = key_pix[o].astype(np.float64) * 100.0 + t[o]
    t_o = t[o]
    if burst_gap is not None:
        new = np.ones(len(o), bool)
        new[1:] = (key_pix[o][1:] != key_pix[o][:-1]) | (np.diff(t_o) > burst_gap)
        stamp = t_o[np.maximum.accumulate(np.where(new, np.arange(len(o)), 0))]
    else:
        stamp = t_o
    ev = order[idx]                                         # original indices of the sampled events
    xe, ye, te, pe = x[ev].astype(np.int64), y[ev].astype(np.int64), t[ev], (p[ev] > 0)
    # reference time of the event itself (its burst onset with burst_gap)
    ke = ((ye * WIDTH + xe) * 2 + pe).astype(np.float64)
    te_ref = stamp[np.clip(np.searchsorted(key, ke * 100.0 + te, side="left"), 0, len(key) - 1)]
    sxx = np.zeros(len(ev)); sxy = np.zeros(len(ev)); syy = np.zeros(len(ev))
    sxt = np.zeros(len(ev)); syt = np.zeros(len(ev)); n = np.zeros(len(ev), int)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            xq, yq = xe + dx, ye + dy
            inside = (xq >= 0) & (xq < WIDTH) & (yq >= 0) & (yq < HEIGHT)
            kq = ((yq * WIDTH + xq) * 2 + pe).astype(np.float64)
            j = np.searchsorted(key, kq * 100.0 + te, side="left") - 1     # last event of q strictly before t_e
            jj = np.clip(j, 0, len(key) - 1)
            t_last = key[jj] - kq * 100.0
            valid = inside & (j >= 0) & (t_last <= te) & (t_last > te - horizon)
            d = np.where(valid, stamp[jj] - te_ref, 0.0)
            sxx += valid * dx * dx; sxy += valid * dx * dy; syy += valid * dy * dy
            sxt += dx * d; syt += dy * d; n += valid
    det = sxx * syy - sxy * sxy
    ok = (n >= 4) & (det > 1e-9)
    a = np.where(ok, (syy * sxt - sxy * syt) / np.where(ok, det, 1), 0.0)
    b = np.where(ok, (sxx * syt - sxy * sxt) / np.where(ok, det, 1), 0.0)
    # Arrival times grow along the motion, so the edge moves along +grad T with speed 1 / |grad T|
    g2 = a * a + b * b
    vx = np.where(ok & (g2 > 0), a / np.where(g2 > 0, g2, 1), np.nan)
    vy = np.where(ok & (g2 > 0), b / np.where(g2 > 0, g2, 1), np.nan)
    return vx, vy, n


def speed_exponent(speed, value):
    """Log-log slope of the per-speed-bin median of value against the bin's median speed."""
    mids, meds = [], []
    for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        sel = (speed >= lo) & (speed < hi) & (value > 0)
        if sel.sum() >= MIN_EVENTS_PER_BIN:
            mids.append(np.median(speed[sel]))
            meds.append(np.median(value[sel]))
    exponent = float(np.polyfit(np.log(mids), np.log(meds), 1)[0]) if len(mids) >= 3 else float("nan")
    return dict(exponent=exponent, speed=[float(m) for m in mids], median=[float(m) for m in meds])


# ----------------------------------------------------------------------------------------------------------------
# Statistics

def fit_scale(v_est, v_true):
    """Robust scale k of v_est = k v_true: median ratio over events with a clear true signal."""
    strong = np.abs(v_true) >= np.median(np.abs(v_true))
    return float(np.median(v_est[strong] / v_true[strong]))


def summarise(name, v_est, v_true, speed, pol):
    nonzero = v_est != 0
    rows = []
    for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        sel = (speed >= lo) & (speed < hi) & nonzero
        if sel.sum() < MIN_EVENTS_PER_BIN:
            continue
        same = np.sign(v_est[sel]) == np.sign(v_true[sel])
        log_ratio = np.log(np.abs(v_est[sel][same] / v_true[sel][same]))
        q25, q75 = np.percentile(log_ratio, [25, 75])
        rows.append(dict(speed_lo=float(lo), speed_hi=float(hi), n=int(sel.sum()),
                         speed_mid=float(np.median(speed[sel])),
                         k=fit_scale(v_est[sel], v_true[sel]),
                         sign_agreement=float(same.mean()),
                         log_ratio_iqr=float(q75 - q25),
                         corr=float(np.corrcoef(v_est[sel], v_true[sel])[0, 1])))
    ks = np.array([r["k"] for r in rows])
    mids = np.array([r["speed_mid"] for r in rows])
    valid = ks > 0
    beta = float(np.polyfit(np.log(mids[valid]), np.log(ks[valid]), 1)[0]) if valid.sum() >= 3 else float("nan")
    overall = fit_scale(v_est[nonzero], v_true[nonzero])
    per_polarity = {}
    for s, label in [(1, "ON"), (-1, "OFF")]:
        sel = nonzero & (pol == s)
        k_pol = fit_scale(v_est[sel], v_true[sel])
        per_polarity[label] = dict(k=k_pol, c=float(1 / k_pol) if k_pol > 0 else float("nan"))
    return dict(name=name, beta=beta, k=overall, c=float(1 / overall) if overall > 0 else float("nan"),
                k_spread_across_speeds=float(ks[valid].max() / ks[valid].min()) if valid.any() else float("nan"),
                sign_agreement=float((np.sign(v_est[nonzero]) == np.sign(v_true[nonzero])).mean()),
                coverage=float(nonzero.mean()), per_polarity=per_polarity, bins=rows)


# ----------------------------------------------------------------------------------------------------------------
# Plot: small multiples, one estimator per panel (single series, no legend; the panel title names it)

SURFACE, TEXT, TEXT_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def plot(results, path):
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.7 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.set_visible(False)
    for ax, r in zip(axes.flat, results):
        ax.set_visible(True)
        mids = np.array([b["speed_mid"] for b in r["bins"]])
        ks = np.array([b["k"] for b in r["bins"]])
        rel = ks / r["k"]
        ax.axhline(1.0, color=TEXT_2, lw=1, zorder=1)
        ax.plot(mids, rel, color=SERIES, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.scatter(mids, rel, s=36, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(r["name"], color=TEXT, fontsize=10, loc="left")
        ax.text(0.02, 0.04, f"β = {r['beta']:+.2f}", transform=ax.transAxes, color=TEXT_2, fontsize=9)
        ax.grid(True, which="major", color=GRID, lw=1)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("image speed |u| (px/s)", color=TEXT_2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("k(speed) / k", color=TEXT_2, fontsize=9)
    fig.suptitle("Scale of each V estimator against brightness-constancy V, by image speed (flat = right time unit)",
                 color=TEXT, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


# ----------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("res/shapes_rotation"))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/v_units"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1_000_000, help="events evaluated (random subset)")
    ap.add_argument("--frame-window", type=float, default=0.002, help="max |t_event - t_frame| in s")
    ap.add_argument("--min-gradient", type=float, default=0.1, help="min |grad log I| in log units / px")
    ap.add_argument("--border", type=int, default=5, help="ignore pixels this close to the image border")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t, x, y, p = load_events(args.data / "events.txt", args.out / "events_cache.npz")
    calib = load_calibration(args.data / "calib.txt")
    imu = np.loadtxt(args.data / "imu.txt")
    frame_t, frame_grad = frame_gradients(args.data)
    print(f"{len(t)} events, {len(frame_t)} frames, {len(imu)} IMU samples")

    # IMU-event time offset: lag that best aligns the event rate with |omega|
    edges = np.arange(1.0, t[-1], 0.005)
    rate = np.histogram(t, edges)[0].astype(float)
    centres = 0.5 * (edges[1:] + edges[:-1])
    speed_imu = np.linalg.norm(imu[:, 4:7], axis=1)
    lags = np.arange(-0.05, 0.0501, 0.001)
    corr = [np.corrcoef(rate, np.interp(centres + lag, imu[:, 0], speed_imu))[0, 1] for lag in lags]
    lag = float(lags[int(np.argmax(corr))])
    print(f"IMU lag maximising corr(event rate, |omega|): {lag * 1000:.0f} ms (corr {max(corr):.3f})")

    # Translation (mocap), for the record: the experiment assumes rotation only
    gt = np.loadtxt(args.data / "groundtruth.txt")
    v_trans = np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1) / np.diff(gt[:, 0])
    print(f"mocap translation speed: median {np.median(v_trans):.3f} m/s, 95th pct {np.percentile(v_trans, 95):.3f} m/s")

    # Estimators need per-pixel histories: sort by pixel, keeping time order
    pix = y.astype(np.int64) * WIDTH + x
    order = np.argsort(pix, kind="stable")
    t_s, p_s, pix_s = t[order], p[order], pix[order]
    gap_any = previous_gap(pix_s, t_s)
    order_pol = np.argsort(pix * 2 + (p > 0), kind="stable")
    gap_same = np.empty(len(t))
    gap_same[order_pol] = previous_gap((pix * 2 + (p > 0))[order_pol], t[order_pol])
    gap_same_s = gap_same[order]

    # Sample events near a frame and inside the image
    nearest = np.clip(np.searchsorted(frame_t, t_s), 1, len(frame_t) - 1)
    nearest = np.where(np.abs(frame_t[nearest - 1] - t_s) < np.abs(frame_t[nearest] - t_s), nearest - 1, nearest)
    xs, ys = x[order].astype(float), y[order].astype(float)
    candidate = (np.abs(frame_t[nearest] - t_s) < args.frame_window) & (t_s > 1.0)
    candidate &= (xs >= args.border) & (xs < WIDTH - args.border) & (ys >= args.border) & (ys < HEIGHT - args.border)
    idx = np.flatnonzero(candidate)
    if len(idx) > 3 * args.samples:
        idx = np.sort(rng.choice(idx, 3 * args.samples, replace=False))

    # Ground truth at the sampled events. The gradient is read where the edge was at frame time
    # (motion-compensated, bilinear), not at the event pixel.
    omega = np.stack([np.interp(t_s[idx] + lag, imu[:, 0], imu[:, 4 + k]) for k in range(3)], -1)
    ux, uy = rotational_flow(xs[idx], ys[idx], omega, calib)
    dt_frame = t_s[idx] - frame_t[nearest[idx]]
    gx, gy = sample_gradient(frame_grad, nearest[idx], xs[idx] - ux * dt_frame, ys[idx] - uy * dt_frame)
    keep = np.hypot(gx, gy) >= args.min_gradient
    idx, omega, ux, uy, gx, gy = idx[keep], omega[keep], ux[keep], uy[keep], gx[keep], gy[keep]
    if len(idx) > args.samples:
        pick = np.sort(rng.choice(len(idx), args.samples, replace=False))
        idx, omega, ux, uy, gx, gy = idx[pick], omega[pick], ux[pick], uy[pick], gx[pick], gy[pick]
    print(f"{candidate.sum()} events near frames, evaluating {len(idx)} on clear gradients")
    v_true = -(gx * ux + gy * uy)
    # Bin by the event pixel's own image speed: |omega| alone mixes axes (roll moves pixels far less per rad/s)
    speed = np.hypot(ux, uy)
    pol = p_s[idx].astype(float)

    # Diagnostics independent of the estimators: exponents of median value vs image speed (1 = proportional)
    burst = gap_same_s[idx] < 0.02
    diagnostics = dict(
        gradient_vs_speed=speed_exponent(speed, np.hypot(gx, gy)),
        v_true_vs_speed=speed_exponent(speed, np.abs(v_true)),
        normal_speed_vs_speed=speed_exponent(speed, np.abs(gx * ux + gy * uy) / np.hypot(gx, gy)),
        burst_rate_vs_speed=speed_exponent(speed[burst], 1.0 / gap_same_s[idx][burst]),
        burst_rate_vs_normal_speed=speed_exponent(np.abs(gx * ux + gy * uy)[burst] / np.hypot(gx, gy)[burst],
                                                  1.0 / gap_same_s[idx][burst]),
    )
    # Time-surface normal flow: speed and direction against the true normal flow. Labels: burst gap, horizon
    # (the horizon sets the slowest measurable speed, about radius / horizon).
    gn = np.hypot(gx, gy)
    un_true = (gx * ux + gy * uy) / gn                       # signed normal speed along the frame gradient
    for label, gap, horizon in [("last event, 30 ms", None, 0.03), ("onset 5 ms, 30 ms", 0.005, 0.03),
                                ("onset 10 ms, 30 ms", 0.01, 0.03), ("onset 10 ms, 100 ms", 0.01, 0.1),
                                ("onset 20 ms, 100 ms", 0.02, 0.1), ("onset 20 ms, 200 ms", 0.02, 0.2)]:
        vx_ts, vy_ts, n_ts = time_surface_flow(idx, x, y, p, t, order, horizon=horizon, burst_gap=gap)
        s_ts = np.hypot(vx_ts, vy_ts)
        fitted = np.isfinite(s_ts) & (s_ts < 1e5) & (s_ts > 0)
        cos_dir = ((vx_ts * ux + vy_ts * uy) / (s_ts * np.hypot(ux, uy)))[fitted]
        ratio = s_ts[fitted] / np.abs(un_true[fitted])
        bins = [dict(lo=float(lo), hi=float(hi), n=int(sel.sum()), median_ratio=float(np.median(ratio[sel])))
                for lo, hi in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:])
                for sel in [(np.abs(un_true[fitted]) >= lo) & (np.abs(un_true[fitted]) < hi)]
                if sel.sum() >= MIN_EVENTS_PER_BIN]
        diagnostics[f"time_surface ({label})"] = dict(
            exponent=speed_exponent(np.abs(un_true[fitted]), s_ts[fitted])["exponent"],
            fitted_fraction=float(fitted.mean()), median_ratio=float(np.median(ratio)),
            log_ratio_iqr=float(np.subtract(*np.percentile(np.log(ratio), [75, 25]))),
            direction_cos_median=float(np.median(cos_dir)), ratio_by_normal_speed=bins)
        d = diagnostics[f"time_surface ({label})"]
        print(f"time surface ({label:19s}): exponent {d['exponent']:.2f}, median ratio {d['median_ratio']:.2f}, "
              f"IQR(log) {d['log_ratio_iqr']:.2f}, direction cos {d['direction_cos_median']:.2f}, "
              f"fitted {d['fitted_fraction']:.0%}; ratio by |u_n|: " + " ".join(f"{b['median_ratio']:.2f}" for b in bins))
    print("speed exponents (log-log slope vs image speed; 1 = proportional):",
          {k: round(v["exponent"], 2) for k, v in diagnostics.items() if not k.startswith("time_surface")})

    estimators = {
        "constant (current: p · 10)": pol * 10.0,
        "1 / Δt (previous event, any polarity)": np.where(np.isfinite(gap_any[idx]), pol / gap_any[idx], 0.0),
        "1 / Δt (previous event, same polarity)": np.where(np.isfinite(gap_same_s[idx]), pol / gap_same_s[idx], 0.0),
    }
    for tau in LEAKY_TAUS:
        estimators[f"leaky rate, τ = {tau * 1000:g} ms"] = leaky_rate(idx, pix_s, t_s, p_s.astype(float), tau)
    estimators[f"paper bins, {BIN_LENGTH * 1000:g} ms"] = bin_rate(idx, pix_s, t_s, p_s, BIN_LENGTH)

    results = [summarise(name, v, v_true, speed, pol) for name, v in estimators.items()]

    meta = dict(seed=args.seed, samples=int(len(idx)), near_frames=int(candidate.sum()), imu_lag_s=lag,
                diagnostics=diagnostics,
                frame_window_s=args.frame_window, min_gradient=args.min_gradient,
                translation_speed_median=float(np.median(v_trans)),
                translation_speed_p95=float(np.percentile(v_trans, 95)),
                note="k in V_est = k V_true with c = 1 (events/s); c = 1/k is in the frame's log units, which "
                     "include the unknown APS response curve.")
    (args.out / "results.json").write_text(json.dumps(dict(meta=meta, estimators=results), indent=2))
    plot(results, args.out / "k_vs_speed.png")

    print(f"\n{'estimator':42s} {'beta':>6s} {'k max/min':>9s} {'sign':>6s} {'coverage':>8s} {'c (ON/OFF)':>16s}")
    for r in results:
        pp = r["per_polarity"]
        print(f"{r['name']:42s} {r['beta']:+6.2f} {r['k_spread_across_speeds']:9.2f} {r['sign_agreement']:6.1%} "
              f"{r['coverage']:8.1%} {pp['ON']['c']:7.3f}/{pp['OFF']['c']:.3f}")


if __name__ == "__main__":
    main()
