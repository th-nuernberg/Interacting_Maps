# G → F: brightness constancy in sign (shapes_rotation)

2026-11-06 · C++: `--polaritySign`, `timing::directionProbability` · data: `--dumpEvents 20` runs · results: R.txt of `output/sign_{off,gate,weight}`

## Question

Brightness constancy, V = −G·F, couples G and F. V's size is not available from events ([v-units](v-units.md)), but its sign is the event's polarity. Can the relation in sign, polarity = sign(−G·F), improve F, and through it R?

## Is G informative about F? (per-event dump, current defaults)

Each timing measurement N (direction n, speed s) is compared with the gyro's true flow u, split by whether it agrees with polarity and G:

| N | share | n points against the true motion | N wrong (flipped, or speed off by > 2×) |
|---|---|---|---|
| agrees: sign(−G·n) = p | 84 % | 3.7 % | 15 % |
| disagrees | 16 % | **39 %** | **53 %** |

| \|G\| (steps/px) | share | wrong if it agrees | wrong if it disagrees |
|---|---|---|---|
| < 0.4 | 8 % | 24 % | 24 % |
| 0.4–0.8 | 40 % | 10 % | 54 % |
| 0.8–1.5 | 39 % | 16 % | 71 % |
| > 1.5 | 13 % | 22 % | 73 % |

Below about 0.4–0.5 steps/px, G carries no information. That matches **G's noise, which can be derived:** I keeps σ²·rate·τ/2 ≈ 0.5 steps² of error (σ² = 1.9 steps² per event, 8.8 events/s, τ = 0.06 s; [forgetting](forgetting.md)), and G = half the difference of two pixels, so σ_G = √(σ²·rate·τ/4) ≈ 0.5 steps/px.

## The relation

The probability that a measurement's direction is right, given polarity and G (`timing::directionProbability`):

- q = Φ(\|G·n\| / σ_G): the probability that G's sign along n is right;
- P(n right | agrees) = q(1 − π) / (q(1 − π) + (1 − q)π);
- P(n right | disagrees) = (1 − q)(1 − π) / ((1 − q)(1 − π) + qπ);
- π = 0.1: the share of flipped directions among all measurements (measured: 9.5 %).

Two ways to apply it:
- **`weight`:** the measurement's information is multiplied by P.
- **`gate`:** a measurement that disagrees is dropped where \|G·n\| > σ_G.

## Results (R against the gyro, 60 s, events only, frames with \|ω\| > 1 rad/s)

| `--polaritySign` | cos median | cos mean | frames cos < 0.9 | cos < 0.5 | \|ω\| ratio | per-axis corr |
|---|---|---|---|---|---|---|
| off | 0.9916 | 0.9380 | 12.27 % | 2.87 % | 0.995 [0.86, 1.14] | 0.956 / 0.963 / 0.954 |
| gate | 0.9915 | 0.9374 | 12.27 % | 2.91 % | 0.999 [0.87, 1.15] | 0.956 / 0.963 / 0.954 |
| **weight** | **0.9918** | **0.9399** | **11.97 %** | **2.67 %** | 1.000 [0.87, 1.15] | **0.958 / 0.964 / 0.956** |

`weight` is slightly better on every score and is the new default. `gate` gains nothing: dropping helps where G is strong, but the hard cut also removes the many disagreeing measurements that are fine.

## Why the gain is small: R is late, not wrong

R averages about 10 000 measurements, so the minority of bad ones barely matters for a rigid rotation. The frames where R is off (cos < 0.9) have a different cause:

| angular acceleration (rad/s²) | 0–20 | 20–50 | 50–100 | > 100 |
|---|---|---|---|---|
| frames with cos < 0.9 | 0.6 % | 6.2 % | 9.7 % | 24.6 % |

| angular speed (rad/s) | 1–2 | 2–3 | 3–5 | 5–8 | > 8 |
|---|---|---|---|---|---|
| frames with cos < 0.9 | 26.9 % | 18.0 % | 12.4 % | 4.8 % | 1.5 % |

**R matches the gyro best 14 ms later**: shifted by 14 ms, the mean cos rises from 0.940 to 0.982. The spikes come from the estimator's latency, at turning points and in slow phases, where a late estimate points the wrong way. Likely sources:
- the evidence lifetime (an exponential average over 10 ms lags by about 10 ms);
- the onset and plane-fit windows (depression τ = 20 ms, horizon 30 ms).

## Consequences

- **G → F is now part of the network** (`--polaritySign weight`, default): the first relation through which I and G affect F and R. For pure rotation the gain is small. It should matter more where F is local (moving objects, translation), because there a single bad measurement is not averaged away.
- **F → G** (correcting G's sign where F is reliable) is still open.
- **Next for R: its latency.** A measurement describes the motion at the time the edge arrived, not now. Options: date each measurement at its onset and predict to the present with the rotation's rate of change, or shorten the windows where the evidence allows it.
