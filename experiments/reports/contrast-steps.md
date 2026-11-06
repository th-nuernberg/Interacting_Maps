# Separate ON/OFF contrast steps instead of a leak? (shapes_rotation)

2026-11-06 · script: `experiments/contrast_steps.py` · C++: `--contrastSteps`, `--dumpEvents 20` · results: `experiments/results/contrast_steps/`

## Question

OFF events outnumber ON events 1.36 : 1 per pixel, while the scene does not get darker (the frames' mean log I goes from 4.13 to 4.20 over the minute). So most likely c_ON ≈ 1.36 c_OFF ([assumptions](../../docs/assumptions.md), W3 and V7). With equal steps, I drifts, and the inherited decay (τ = 0.1 s) is what keeps it usable. Do ON/OFF steps estimated from the event counts make the decay unnecessary?

## Setup

- **Runs:** five 60 s runs in timing mode, events only, seed 0, no clamps (the default since `e4a33a4`), I in units of c_OFF.
- **Contrast steps:**
  - `equal`: ±1;
  - `global`: ON step = OFF/ON count ratio over all pixels so far;
  - `pixel`: per-pixel count ratio, with 100 pseudo-events of the global ratio.
- **Decay:** each mode with and without the decay.
- **Scores:**
  - I at the event pixels over time;
  - polarity = sign(−G·u_gyro), for ON and OFF events separately;
  - G against the frame gradient (49 739 events near frames).
- R is bit-identical in all runs, because timing mode does not read I or G.

## Results

| | agreement (ON / OFF) | cos(G, frame ∇log I), median | I after 50 s: median, 99 % of \|I\| |
|---|---|---|---|
| **equal + decay** (baseline) | **0.836** (0.832 / 0.838) | **0.923** | −0.9, 3.2 |
| global + decay | **0.839** (0.846 / 0.833) | **0.926** | −0.6, 3.2 |
| equal, no decay | 0.531 (0.528 / 0.533) | 0.128 | **−73**, 276 |
| global, no decay | 0.532 (0.534 / 0.531) | 0.128 | −6, 386 |
| pixel, no decay | 0.539 (0.542 / 0.537) | 0.159 | **−0.5**, 139 |

The global ratio estimate settles within about 1 s: 1.32 after 1 s, 1.36 after 10 s, 1.32 after 60 s (`ratio_estimate.png`).

## Findings

1. **The step ratio removes the systematic drift.** Without decay, the median I after 50 s goes from −73 steps (equal) to −6 (global) and −0.5 (per pixel). This confirms the count-based estimate: with c_ON/c_OFF ≈ 1.3–1.4, each pixel's I returns to where it started.
2. **It doesn't make I usable without the decay.** The *spread* still grows with time, like a random walk: 99 % of \|I\| reaches 139–386 steps after 50 s. G is then dominated by the difference between neighbouring pixels' accumulated errors, not by edges. Polarity agreement is close to chance (0.53) and cos(G, frame) is 0.13–0.16. (The earlier 0.70 without decay came from the [0, 255] clamp, which bounded that walk.)
3. **The error per event is not exactly one threshold.** Each pixel adds about 380 events a minute, and every one carries an error: threshold noise from event to event, mismatch between pixels, noise events, refractory losses. With nothing that forgets, these errors add up. **That is what the decay actually does:** it forgets accumulated error. It isn't only compensating the ON/OFF imbalance.
4. **With the decay, the global ratio helps a little and evens out the polarities.** Agreement goes from 0.836 to 0.839; ON events go from 0.832 to 0.846 and are now slightly ahead of OFF events (0.833). The difference is small, because with τ = 0.1 s, I only covers the last few events at a pixel.

## Consequences

- **Separate steps are right, but they are not a replacement for forgetting.** They stay switchable (`--contrastSteps`, assumption V7) until the real thresholds are known.
- **A grounded form of the decay** would treat I as an estimate whose uncertainty grows with every event, by the variance of one step (threshold noise σ_c), and which falls back towards "unknown" (0) at a rate set by that growth. The τ would then follow from σ_c and the pixel's event rate, not from a chosen 0.1 s. That needs σ_c. It can be estimated from the data, for example from how far I spreads without decay (finding 2), or measured with a threshold calibration.
- **Spatial coupling is not enough on its own here.** I ← G, which should tie I to its neighbours' gradients, does not stop the per-pixel random walk.

## Open

- Estimate σ_c from the random-walk growth of I without decay, and derive the forgetting rate from it.
- Repeat on the simulator (M1), where threshold noise and mismatch are known.
