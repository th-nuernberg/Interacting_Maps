# Rotation from event timing: loop vs steady state (offline prototype, shapes_rotation)

2026-11-02 · script: `experiments/r_prototype.py` · results: `experiments/results/r_prototype/` (`omega_vs_gyro.png`)

## Question

Does the network's F ↔ R loop, with measurements kept as standing constraints, settle into a state that tracks the camera rotation? Does that state equal the direct steady-state formula, and how many trips does the loop need?

## Setup

**Measurements.** Every event gives a normal flow from the burst-onset time surface of its 5 × 5 neighbourhood (burst gap 10 ms, horizon 30 ms; see `v-units.md`): speed s along direction n, and a variance σ² from the plane-fit residuals (time floor 0.1 ms). This worked for 96 % of events (median s 464 px/s, median σ_s 27 px/s).

**Standing constraints.** Each pixel keeps its latest measurement, with weight w = e^(−age/τ)/σ².

**Energy.** One term per piece of evidence:

E = Σ_measured w (n·F − s)² + κ Σ_all |F − Jω|²

- J maps the camera's angular velocity ω (rad/s) to pixel flow (px/s), with lens distortion.
- κ = 1/(27 px/s)², set equal to the typical measurement precision. This is open.

**Two solutions, evaluated at 60 Hz:**
- **Steady state:** pixels without measurements follow ω exactly and drop out, so ω* = argmin Σ_measured w′(n·Jω − s)², with w′ = wκ/(w + κ).
- **Loop:** the network's structure. F follows ω at every pixel, then ω is refitted from all pixels' F. In closed form: ω ← ω + A_all⁻¹(b − Hω). It is warm-started, with T trips per window. Its fixed point is ω*.

**The gyro is not used** except to score the result (with a 4 ms IMU lag).

## Results

| τ | Measured pixels per window | Steady state: cos(ω, gyro), median / mean | \|ω\| / \|gyro\| median [IQR] | Per-axis corr | Loop trips per factor e |
|---|---|---|---|---|---|
| **10 ms** | 10 500 | **0.99** / 0.92 | **1.00** [0.86, 1.16] | 0.95 / 0.95 / 0.95 | 92 |
| 30 ms | 22 300 | 0.96 / 0.78 | 0.94 [0.73, 1.22] | 0.85 / 0.84 / 0.84 | 42 |
| 100 ms | 37 000 | 0.79 / 0.51 | 0.74 [0.48, 1.10] | 0.59 / 0.58 / 0.56 | 18 |

Loop with τ = 10 ms, by trips per 1/60 s window:

| Trips | 1 | 3 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|---|
| cos (median) | 0.13 | 0.33 | 0.72 | 0.94 | 0.98 | 0.99 |
| \|ω\| ratio | 0.21 | 0.39 | 0.72 | 0.93 | 0.99 | 1.00 |
| gap to steady state | 0.97 | 0.92 | 0.68 | 0.30 | 0.06 | 0.00 |

## Findings

1. **Event timing alone gives the camera rotation in rad/s.** With τ = 10 ms, the direction matches the gyro (median cos 0.99) and so does the magnitude (ratio 1.00), with no scale fitted. The current network gets cos ≈ 0.02 and has no units.
2. **The loop settles into exactly the steady state** (the gap goes to 0). So the assumption that information may take many trips holds. It needs about 100–300 trips per 1/60 s window to get there.
3. **Why so many trips.** Each ω-step refits from all 43 200 pixels, while only about 10 500 carry a measurement. Most of the fit is pixels repeating the current ω, so every trip moves ω only a fraction of the way. This is the dilution from the diagnosis, now measured: about 92 trips per factor e.
4. **Short lifetimes are better here** (τ = 10 ms beats 30 and 100 ms). Old measurements describe a rotation that has already changed. τ trades noise against lag and depends on how fast the motion changes.
5. **The mean cos (0.92) is below the median (0.99)**, so a few windows are poor. Which ones (slow segments? translation?) is still to be examined.

## For the C++ network

- **The trip budget is set by how fast unmeasured pixels follow R.** At the current 644/s, a trip takes about 1.6 ms, which allows about 10 trips per window: cos 0.72 and a ratio of 0.72, i.e. the estimate lags.
- **The fixed point doesn't depend on the trip speed**, so the loop can be made faster without changing what it converges to:
  - let unmeasured pixels follow R instantly (that's their exact steady state, since they have no other term), or
  - refit R from the measured pixels' weights (H instead of A_all). This is the same loop, preconditioned; it converges in about one trip.
- **Measurements must stay in the loop** as fading constraints. A one-off kick at the event vanishes before the loop has used it.

## Open

- κ (model precision) and τ (measurement lifetime) are open parameters. Sweep them on the simulator (M1), and check translation using the mocap.
- Look at the poor windows (mean vs median cos).
- Feed the network's G into the sign and direction checks of the proposal (this prototype uses timing only).
