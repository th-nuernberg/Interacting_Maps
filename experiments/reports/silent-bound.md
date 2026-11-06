# What should a silent pixel do to G? (shapes_rotation)

2026-11-06 · C++: `--silentRule`, `--refractoryTime`, `--silentSlack`, `--dumpEvents 20` · analysis: `experiments/polarity_check.py` · results: `experiments/results/silent_bound/`, `experiments/results/silent_bound_gradI/`

## Question

Between events, G was relaxed with two inherited rates (22.3/s): towards ∇I, and towards "no component along F" (V = 0). An event camera's silence says something more specific. The change of log I since the pixel's last event is still between the OFF threshold (−1 step) and the ON threshold (+c_ON/c_OFF). With brightness constancy, ΔI = −(G·F)·Δt, so G's component along F would be bounded by the thresholds divided by the distance travelled, |F|·Δt. That bound has no free parameter. Does it replace the rates?

## Setup

- Timing mode, 60 s, events only, current defaults (per-pixel contrast steps, τ = 0.06 s).
- **Refractory time, measured:** the shortest interval between two events at one pixel is 1.2 ms (0.001 % quantile 1.22 ms; only 2 per million are shorter, all duplicates). Silence counts from 1.2 ms after the last event.
- **Variants:**
  - `rate` (the former default);
  - `bound`, hard;
  - `bound` with slack 0.5 and 1.4 steps (the measured error per event);
  - `bound` without the refractory skip;
  - G = ∇I with no silence rule (`gradient`).
- **Scores:** polarity = sign(−G·u_gyro) (as in [polarity-check](polarity-check.md)), and G against the frame gradient for 49 739 events near frames.

## Results

| rule | polarity agreement | cos(G, frame ∇log I), median | silent > 0.1 s: agreement |
|---|---|---|---|
| `rate` (former default) | **0.850** | **0.934** | 0.89–0.91 |
| **`gradient`** (G = ∇I when read) | **0.850** | — (identical bins) | 0.89–0.91 |
| `bound`, hard | 0.752 | 0.809 | 0.50–0.58 |
| `bound`, slack 0.5 | 0.765 | 0.820 | 0.51–0.61 |
| `bound`, slack 1.4 | 0.779 | 0.835 | 0.53–0.66 |
| `bound`, no refractory | 0.750 | 0.807 | 0.50–0.58 |

`gradient` and `rate` agree in every silence bin to within 0.001. The bound's results depend on how long the pixel was silent: the longer the silence, the more |G| is cut and the closer agreement gets to chance (`by_time_since_previous_event.png`).

## Findings

1. **The bound is wrong where G is used.** Silence constrains G *during* the quiet interval, and G was indeed small then. But G is read at a pixel at the moment an edge *arrives*: the neighbours have fired and ∇I already contains the arriving edge. Applying the bound from the long preceding silence to the current ∇I cuts exactly the gradient that causes the event. The longer the silence, the stronger the cut, hence chance level after 0.1 s or more. Slack only softens this, and the refractory skip makes no difference.
2. **The bound never changes the sign of G·F with the stored F**, since it only scales the component along F. Agreement with the stored F stays at 0.847 in every variant. The loss against the gyro flow comes from removing the component that carries the edge.
3. **The inherited rates have no measurable effect in timing mode.** G = ∇I, computed whenever G is read, gives the same result as the rate relaxation in every bin. With I forgetting at the derived τ = 0.06 s ([forgetting](forgetting.md)), stale gradients fade on their own, and G needs no relaxation of its own.

## Consequences

- **New default `--silentRule gradient`:** G is ∇I when read, its definition, with no parameters. This replaces the two inherited 22.3/s rates. Bit-identical R; same G scores.
- `rate` and `bound` remain switchable. A correct use of silence would need G's *history* (what G was during the silence) or a prediction of the arriving edge, for example a constraint on the change predicted by advecting I with F. That is a candidate for later, best tested on the simulator.
- **Measured refractory time: 1.2 ms** (assumption W12).
