# How much error does each event add to I, and how fast must I forget? (shapes_rotation)

2026-11-06 · script: `experiments/forgetting.py` · C++: `--decayTime`, `--contrastSteps`, `--dumpEvents 20` · results: `experiments/results/forgetting/`

## Question

Without the decay, I spreads like a random walk even with ON/OFF contrast steps ([contrast-steps](contrast-steps.md)). The inherited decay time of 0.1 s works, but it has no justification. How much error does one event add, and what forgetting time follows from it?

## Method

1. **Error per event.** Integrate the events per pixel exactly as the network does without decay: I in units of c_OFF, with equal, global or per-pixel (causal) contrast steps. Then fit the spread of I over pixels as a function of the pixel's number of events N, Var(I) = a + σ²N + qN², for N from 20 to 2000.
   - Independent errors per event give the linear term σ².
   - A systematic error per pixel (a wrong step ratio) gives the quadratic term q.
   - The true scene contrast stays bounded (a).
2. **Forgetting time.** A leak with time constant τ keeps σ²·rate·τ/2 of error variance per pixel. G reads the difference of two neighbours, which carries σ²·rate·τ. Forgetting before that reaches one step² gives **τ\* = 1/(σ²·rate)**.
3. **Check.** C++ runs sweeping τ = 0.01–1 s with equal and per-pixel steps, scored by polarity agreement sign(−G·u_gyro) = p.

## Results

**Error growth** (`variance_growth.png`):

| contrast steps | σ² (steps² per event) | quadratic part at N = 1000 | linear part at N = 1000 |
|---|---|---|---|
| equal | 0.42 | 15 757 | 417 |
| global | 0.60 | 21 238 | 603 |
| **per pixel** | **1.91** | **414** | 1 910 |

- With one shared ratio, the **quadratic term dominates**: each pixel has its own c_ON/c_OFF, and a shared ratio leaves a systematic drift per pixel. Only per-pixel steps remove it. So the spread of the per-pixel ratio (IQR 1.22–1.53) is real pixel mismatch, not counting noise.
- What remains is close to a random walk, with **σ ≈ 1.4 steps per event**. That is far more than threshold noise on an ideal pixel. A burst's event count only measures the edge's contrast on average: the pixel's bandwidth changes how many events an edge produces (compare the speed^0.24 burst rate in [v-units](v-units.md)).
- For N below about 10, Var(I) levels off at 6–8 steps². That is the real scene contrast (a few steps per edge).

**Derived forgetting time:** active pixels fire at a median of 8.8 events/s (IQR 6.8–11.1). τ\* = 1/(1.91 · rate) = **0.06 s** (IQR 0.05–0.08 s).

**Sweep** (`decay_sweep.png`):

| τ (s) | 0.01 | 0.03 | **0.06** | 0.1 (inherited) | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| equal steps | 0.815 | 0.841 | 0.841 | 0.836 | 0.796 | 0.698 |
| per-pixel steps | 0.814 | 0.845 | **0.850** | 0.848 | 0.824 | 0.750 |

## Findings

1. **The derived τ\* = 0.06 s is the optimum of the sweep.** Per-pixel steps peak there at 0.850; equal steps are flat between 0.03 and 0.06 s. The inherited 0.1 s was close: a sensible empirical choice, now with a derivation.
2. **Per-pixel steps help at every τ,** and most when I remembers longer, because there the systematic drift of a shared ratio dominates (0.750 vs 0.698 at 1 s).
3. **Too short also hurts** (0.81 at 10 ms): I then forgets the edge before its neighbours' events arrive, so G loses the edge.
4. **New defaults:** `--contrastSteps pixel` and `--decayTime 0.06`. Polarity agreement goes from 0.836 to 0.850; R is unchanged (timing mode does not read I or G).

## Limits

- σ² and the event rate come from this one recording. τ\* is a formula: a sensor in a darker scene (fewer events, more noise) needs another value. The next step would be per-pixel forgetting with each pixel's own rate, τ_p = 1/(σ²·rate_p), with σ² estimated online.
- "One step² of error" is a criterion. The sweep confirms it here, but the optimum is flat between 0.03 and 0.1 s.
- The error model treats events as independent, while errors are probably correlated within a burst (bandwidth). That doesn't change the fit's slope much, but it is why σ is larger than threshold noise.
