# Onset rules: when did the edge arrive? (shapes_rotation)

2026-11-06 · script: `experiments/onset_rules.py` · shared maps and relations: `experiments/event_maps.py` · results: `experiments/results/onset_rules/` (`speed_ratio_by_rule.png`)

## Question

An edge crossing a pixel triggers a burst of events. The O map should hold the time the edge *arrived*. Which rule turns a pixel's events into that arrival time best?

## Rules compared

All rules share one interface (`rule(index) → stamps, weights`) and feed the same O → N relation (plane fit over the 5 × 5 neighbourhood, horizon 30 ms).

| Rule | Arrival time O | Biological counterpart |
|---|---|---|
| latest event | time of the pixel's latest event | — |
| hard gap 10 ms | first event of the current burst; a gap > 10 ms starts a new one | refractory recovery as a step |
| depression τ, U | O ← O + r (t − O); r = recovered resource, r ← 1 − (1 − r)e^(−Δt/τ) before an event, r ← (1 − U)r after | short-term synaptic depression (Tsodyks–Markram) |

The hard gap is the step-shaped limit of depression. The depression rule is checked against its limits: τ → 0 gives the latest event, τ → ∞ the first onset.

## Scores

- **N:** normal flow at 1 M events near DAVIS frames, against the true normal flow (gyro plus frame gradient; see `v-units.md`). Speed ratio estimated/true (median and spread of its log) and direction cosine.
- **ω:** camera angular velocity from the standing constraints of all 23 M events, against the gyro. Steady state, κ = ∞, J with lens distortion, measurement lifetime 10 ms; see `r-prototype.md`.

## Results

| Rule | N: speed ratio (median) | N: IQR(log) | N: direction cos | ω: cos median / mean | ω: \|ω\| ratio [IQR] | ω: per-axis corr |
|---|---|---|---|---|---|---|
| latest event | 1.75 | 0.76 | 0.79 | 0.989 / 0.914 | 0.98 [0.85, 1.13] | 0.94 / 0.95 / 0.94 |
| **hard gap 10 ms** | **1.15** | **0.47** | 0.80 | 0.991 / 0.933 | 0.98 [0.85, 1.12] | 0.955 / 0.960 / 0.949 |
| depression 2 ms | 1.58 | 0.67 | 0.79 | 0.990 / 0.920 | 0.99 [0.86, 1.14] | 0.945 / 0.953 / 0.941 |
| depression 5 ms | 1.39 | 0.59 | 0.80 | 0.991 / 0.931 | 1.01 [0.88, 1.15] | 0.951 / 0.958 / 0.947 |
| **depression 10 ms** | 1.28 | 0.54 | 0.80 | **0.992 / 0.937** | **1.01** [0.88, 1.15] | 0.957 / 0.961 / 0.953 |
| **depression 20 ms** | 1.20 | 0.54 | 0.79 | **0.992 / 0.938** | 1.00 [0.86, 1.14] | **0.956 / 0.963 / 0.954** |
| depression 40 ms | 1.12 | 0.63 | 0.76 | 0.990 / 0.935 | 0.93 [0.77, 1.09] | 0.942 / 0.952 / 0.946 |
| depression 10 ms, U 0.5 | 1.48 | 0.58 | 0.79 | 0.990 / 0.926 | 1.00 [0.86, 1.15] | 0.948 / 0.955 / 0.946 |

## Findings

1. **Any burst handling helps over the latest event.** It cuts the speed overestimate at medium speeds roughly in half (see the plot).
2. **For single normal-flow measurements, the hard gap is best:** smallest bias and spread (ratio 1.15, IQR 0.47).
3. **For rotation, depression with τ = 10–20 ms is best, by a small margin:** mean cos 0.938 vs 0.933, per-axis corr 0.963 vs 0.960, and |ω| ratio 1.00 vs 0.98. The differences are small relative to the spread over time.
4. **Depression has a sweet spot.** Short τ behaves like the latest event; long τ (40 ms) merges separate edges and loses magnitude (ratio 0.93).
5. **U < 1 (keeping resource after an event) is slightly worse** than U = 1 here.
6. **Slow speeds (below about 100 px/s) stay overestimated for every rule.** That's the horizon limit (about radius/horizon = 67 px/s) plus translation in the recording, not the onset rule.

**Conclusion.** Soft onset by depression (τ ≈ 10–20 ms, U = 1) is at least as good as the hard gap for rotation. It is biologically grounded and needs no threshold. The hard gap gives slightly cleaner single measurements. Both are O(1) per event with one or two numbers per pixel. Keep the onset rule swappable in the C++ port, and decide on the simulator (M1).

Correction during this run: an earlier definition moved the arrival time by the resource *used* (U·r) instead of the resource *recovered* (r). For U < 1 that never resets after a long silence and gave nonsense. U = 1 is unaffected; U = 0.5 was rerun with the corrected rule.
