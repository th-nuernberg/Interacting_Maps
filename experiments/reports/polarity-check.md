# Does the network's G predict event polarity? (shapes_rotation)

2026-11-06 · script: `experiments/polarity_check.py` · C++ dump: `--dumpEvents 20` (src/main.cpp) · results: `experiments/results/polarity_check/`

## Question

Brightness constancy, V = −G·u, predicts the sign of every event: polarity = sign(−G·u). Timing mode now gives the motion u in px/s, so the open question for the F ↔ G coupling is whether **G** is good enough to be used at all. Where its errors come from decides what to fix first.

## Setup

- **Runs:** C++ network, timing mode, events only, 0–60 s, seed 0. One run with the defaults, and one with `--decay false` (I no longer decays to 128).
- **What is recorded:** every 20th event, giving 1.16 M events per run. The maps are read at the event pixel just before the event's own I/G update: G, the stored F, M·R, the gyro flow M·R_gyro, the event's normal flow N, I on the 3 × 3 stencil, and the time since the pixel's previous event.
- **Frame reference:** 49 739 of these events lie near a DAVIS frame and on a clear gradient. For them, the true ∇log I and image motion u come from `v_units.ground_truth_events`, joined by (time, x, y).

## Results

| | default (decay τ = 0.1 s) | no decay |
|---|---|---|
| **sign agreement, u = gyro flow** | **0.836** | 0.701 |
| u = M·R (network) | 0.826 | 0.695 |
| u = stored F | 0.834 | 0.701 |
| u = N (this event) | 0.832 | 0.710 |
| same events, **frame gradient** with gyro flow | **0.959** | 0.959 |
| same events, network G with gyro flow | 0.853 | 0.714 |
| cos(G, frame ∇log I), median / share > 0 | **0.92 / 91 %** | 0.75 / 76 % |
| \|cos(G, N)\| median (random directions: 0.71) | 0.92 | 0.82 |
| G = 0 at the event | 0.1 % | 7 % |
| events whose I stencil touches the clamp (0 or 255) | **0.0 %** | 78 % (74 % at 0) |

Plots:
- `sign_agreement.png`: the first rows of the table;
- `direction_of_G.png`: G against N and against the frame gradient;
- `by_time_since_previous_event.png`: agreement and |G| by the time since the pixel's previous event;
- `clamping_of_I.png`: the clamping over time.

## Findings

1. **G carries real information.** It points the right way (median cos 0.92 with the true gradient) and predicts the polarity for 84 % of events. The frame gradient on the same events gets 96 %, so G is wrong about 3.5× as often as the frames (15 % vs 4 %).
2. **The errors are in G, not in the motion.** Gyro flow, network R, stored F and the event's own N all give the same agreement to within 1 %. The timing-based F is as good as the gyro for this purpose, so improvements belong in the I → G path.
3. **|G| works as a confidence measure.** In the lowest decile (|G| < 4 I units/px, less than half an event's step of 10) agreement is 0.57, close to chance. From the 4th decile up it is 0.85–0.9. Typical |G| is 7–9, so G rests on about one event's difference between neighbours.
4. **Freshness matters, but in an unexpected way.** Events on pixels that were silent for more than 0.1 s agree best (0.89–0.90). Events 5–10 ms after the previous one agree worst (0.75). That is the tail of a burst, when the edge has partly passed and the stencil's neighbours are in mixed states. This explanation is a hypothesis and hasn't been checked.
5. **Without decay, I drifts into the clamp.** I runs into its lower bound within about 10 s: 74 % of events have a stencil at 0, because OFF events outnumber ON events. Every measure drops to about 0.70. **So the decay is what currently keeps I usable.** It acts as a leak against the ON/OFF imbalance, like adaptation in a photoreceptor. Its τ = 0.1 s is still not justified.
6. **With decay on, the clamps never act.** I never touches 0 or 255 (0.0 %), and |G| reaches at most 68, far below its ±255 clamp. In the default configuration the clamps are dead code, so they can be removed without changing the result.

## Consequences for the F ↔ G work

- A **polarity relation** (sign of the event vs −G·F) is viable, provided its weight follows |G|.
- **Next fixes are in I → G:** justify or sweep the leak τ; possibly balance ON/OFF steps (separate c for ON and OFF) instead of leaking; remove the I and G clamps (they are inactive with decay anyway).
- **Side finding:** `slice_events` includes both ends of each 1 s slice. With float times, about 40 boundary events per minute are processed twice. That is negligible here, but it is a bug, and it is pinned by a test.

## Open

- Sweep the leak τ, and test an ON/OFF-balanced I (no leak) against this baseline.
- Check the hypothesis about burst tails, for example by using G from before the burst began.
- Repeat on the simulator (M1), where ∇log I is exact and there is no translation.
