"""Maps and relations shared by the event experiments, behind small interfaces so they can be swapped.

    events --(onset rule)--> O (arrival-time stamps) --(normal_flow)--> N (n, s, variance) --> F, R, ...

EventIndex     the event stream sorted once by (pixel, polarity), then time; every relation reads it.
Onset rules    callables  rule(index) -> (stamps, weights), one arrival-time stamp per event in index order.
               weights say how much each event counts as news (1 = a new arrival, 0 = more of the same).
normal_flow    the O -> N relation: plane fit to neighbour stamps. It only reads stamps, so any onset rule plugs in.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

WIDTH, HEIGHT = 240, 180


@dataclass
class EventIndex:
    t: np.ndarray          # event times, original order
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray          # -1 / +1
    order: np.ndarray      # original index of each sorted position
    position: np.ndarray   # sorted position of each original event
    key: np.ndarray        # (pixel * 2 + polarity) * 100 + t, sorted; monotone, t < 100 s
    group: np.ndarray      # pixel * 2 + polarity, sorted
    t_sorted: np.ndarray
    gap: np.ndarray        # time since the previous event of the same pixel and polarity (inf for the first)

    @classmethod
    def build(cls, t, x, y, p):
        group = (y.astype(np.int64) * WIDTH + x) * 2 + (p > 0)
        order = np.argsort(group, kind="stable")
        position = np.empty_like(order)
        position[order] = np.arange(len(order))
        g, ts = group[order], t[order]
        gap = np.full(len(t), np.inf)
        same = g[1:] == g[:-1]
        gap[1:][same] = np.diff(ts)[same]
        return cls(t, x, y, p, order, position, g.astype(np.float64) * 100.0 + ts, g, ts, gap)


# ----------------------------------------------------------------------------------------------------------------
# Onset rules: events -> O

class LatestEvent:
    """O = time of the pixel's latest event (no burst handling)."""
    name = "latest event"

    def __call__(self, index):
        return index.t_sorted, np.ones(len(index.t_sorted))


class HardGap:
    """O = onset of the current burst; a gap longer than `gap` starts a new burst."""
    def __init__(self, gap):
        self.gap = gap
        self.name = f"hard gap {gap * 1000:g} ms"

    def __call__(self, index):
        new = index.gap > self.gap
        first = np.maximum.accumulate(np.where(new, np.arange(len(new)), 0))
        return index.t_sorted[first], new.astype(np.float64)


@njit(cache=True)
def _depression(t, gap, tau, use):
    stamp = np.empty(len(t))
    weight = np.empty(len(t))
    o = 0.0
    r = 1.0                               # resource available just after the previous event
    for k in range(len(t)):
        if np.isinf(gap[k]):
            r_before = 1.0
        else:
            r_before = 1.0 - (1.0 - r) * np.exp(-gap[k] / tau)
        # The arrival time moves by how far the pixel has recovered (1 after a long silence: a new arrival);
        # `use` only sets how much of the resource this event consumes.
        if np.isinf(gap[k]):
            o = t[k]
        else:
            o = o + r_before * (t[k] - o)
        r = r_before - use * r_before
        stamp[k] = o
        weight[k] = r_before
    return stamp, weight


class Depression:
    """Soft onset by short-term depression (Tsodyks-Markram, one resource variable per pixel and polarity).

    Each event uses a fraction `use` of the pixel's resource r, which recovers towards 1 with time constant tau:
        r_before = 1 - (1 - r_after_previous) exp(-dt / tau),   r_after = (1 - use) r_before
    r_before is how much the event counts as news. The arrival time moves towards the event by it:
        O <- O + r_before (t - O)
    A fully recovered pixel (long silence) resets O to t; events in quick succession barely move it. The hard gap
    is the limit of a step-shaped recovery.
    """
    def __init__(self, tau, use=1.0):
        self.tau, self.use = tau, use
        self.name = f"depression τ {tau * 1000:g} ms" + (f", U {use:g}" if use != 1.0 else "")

    def __call__(self, index):
        return _depression(index.t_sorted, index.gap, self.tau, self.use)


# ----------------------------------------------------------------------------------------------------------------
# Relation O -> N

def normal_flow(index, stamps, events, radius=2, horizon=0.03, time_floor=1e-4):
    """Normal flow at the given events (original indices) from the arrival-time stamps of their neighbours.

    Fits stamp_q - stamp_e = a dx + b dy to the most recent same-polarity event of every neighbour q within
    `radius` that is at most `horizon` older than the event. Arrival times grow along the motion, so the edge
    moves along n = (a, b) / |(a, b)| with speed s = 1 / |(a, b)|. The variance of s comes from the fit residuals
    (floored at time_floor). Returns n_x, n_y, s, var_s (nan where the fit fails) and the neighbour count.
    """
    xe, ye = index.x[events].astype(np.int64), index.y[events].astype(np.int64)
    te, pe = index.t[events], index.p[events] > 0
    ref = stamps[index.position[events]]
    acc = {k: np.zeros(len(events)) for k in ["xx", "xy", "yy", "xt", "yt", "tt"]}
    n = np.zeros(len(events), int)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            xq, yq = xe + dx, ye + dy
            inside = (xq >= 0) & (xq < WIDTH) & (yq >= 0) & (yq < HEIGHT)
            kq = ((yq * WIDTH + xq) * 2 + pe).astype(np.float64)
            j = np.searchsorted(index.key, kq * 100.0 + te, side="left") - 1     # last event of q before t_e
            jj = np.clip(j, 0, len(index.key) - 1)
            t_last = index.key[jj] - kq * 100.0
            valid = inside & (j >= 0) & (t_last <= te) & (t_last > te - horizon)
            d = np.where(valid, stamps[jj] - ref, 0.0)
            acc["xx"] += valid * dx * dx; acc["xy"] += valid * dx * dy; acc["yy"] += valid * dy * dy
            acc["xt"] += dx * d; acc["yt"] += dy * d; acc["tt"] += d * d; n += valid
    det = acc["xx"] * acc["yy"] - acc["xy"] ** 2
    ok = (n >= 4) & (det > 1e-9)
    sd = np.where(ok, det, 1.0)
    a = (acc["yy"] * acc["xt"] - acc["xy"] * acc["yt"]) / sd
    b = (acc["xx"] * acc["yt"] - acc["xy"] * acc["xt"]) / sd
    g2 = a * a + b * b
    ok &= g2 > 0
    g = np.sqrt(np.where(ok, g2, 1.0))
    res = np.maximum(acc["tt"] - a * acc["xt"] - b * acc["yt"], 0.0)
    sr2 = np.maximum(res / np.maximum(n - 2, 1), time_floor ** 2)
    nx, ny = a / g, b / g
    var_g = sr2 * (nx * nx * acc["yy"] - 2 * nx * ny * acc["xy"] + ny * ny * acc["xx"]) / sd
    s = 1.0 / g
    nan = np.full(len(events), np.nan)
    return (np.where(ok, nx, nan), np.where(ok, ny, nan), np.where(ok, s, nan),
            np.where(ok, var_g * s ** 4, nan), n)


def normal_flow_all(index, stamps, chunk=1_000_000, **kwargs):
    """normal_flow for every event, in chunks."""
    out = [np.full(len(index.t), np.nan) for _ in range(4)]
    for c0 in range(0, len(index.t), chunk):
        e = np.arange(c0, min(c0 + chunk, len(index.t)))
        for o, v in zip(out, normal_flow(index, stamps, e, **kwargs)[:4]):
            o[e] = v
    return tuple(out)
