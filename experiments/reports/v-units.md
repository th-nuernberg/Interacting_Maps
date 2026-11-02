# How should events be read as V? (stage A, shapes_rotation)

2026-11-02 · script: `experiments/v_units.py` · results: `experiments/results/v_units/` (seed 0, 1 M events)

## Question

V = ∂ log I / ∂t needs a time unit, and an event only says "log I changed by ±c here". Which reading of the events gives V in real units, so that F comes out in px/s and R in rad/s?

## Method

Ground truth comes from outside the network:

- image motion u of every event pixel, from the gyro (rotation only; lens distortion included);
- gradient of log intensity ∇log I, from the nearest DAVIS frame (within 2 ms, sampled where the edge was at frame time);
- V_true = −∇log I · u.

Each estimator is fitted as V_est = k · V_true per bin of image speed |u|. A correct time unit gives a k that doesn't depend on speed (exponent β = 0).

Checks of the reference itself:

- the IMU axes match the camera's;
- the IMU lags the events by about 4 ms (from the event rate vs |ω|), and that lag is corrected;
- the true normal speed scales as |u|^0.99, as it should.

## Results

**Per-pixel estimators carry almost no speed information.** Every one of them has about the same exponent as the constant V the code uses now:

| Estimator | β (0 = right unit) | Sign agreement |
|---|---|---|
| constant (current, p · 10) | −0.78 | 96 % |
| 1 / Δt, previous event | −0.88 | 96 % |
| leaky rate, τ = 1 / 3 / 10 / 30 ms | −0.77 / −0.78 / −0.90 / −1.05 | 96–84 % |
| paper bins, 10 ms | −0.80 | 96 % |

**Why:** a pixel's own event timing hardly changes with speed. The burst rate (1/Δt between same-polarity events) scales as |u_n|^0.24, where physics predicts 1. This is most likely the pixel's bandwidth and refractory time. The frame-based reference adds a little bias: blur lowers the gradient as |u|^−0.2. That shifts all β down by about 0.2 but doesn't change the comparison.

**Arrival times across neighbouring pixels do carry speed, in px/s and without c.** Fitting a plane to the burst-onset times of the same-polarity events in a 5 × 5 neighbourhood gives the normal flow directly.

| Time surface (burst gap, horizon) | Direction cos (median) | Speed ratio estimated / true, by \|u_n\| from 20 to 1920 px/s |
|---|---|---|
| last event, 30 ms | 0.79 | 6.8 → 1.07 |
| onset 10 ms, 30 ms | **0.80** | 5.6, 2.6, 1.7, 1.4, 1.3, 1.19, 1.10, 1.07, 1.05, 1.04, 1.03 |
| onset 10 ms, 100 ms | 0.64 | 4.3 … 1.10, 1.02, 1.00, 0.98, 0.98, 0.91 |
| onset 20 ms, 200 ms | 0.13 | breaks down (older, unrelated edges) |

Above about 240 px/s the onset time surface is within 3–19 % of the true speed with no free scale. Below that it overestimates. Two reasons:

- the horizon caps the slowest measurable speed at about radius / horizon (67 px/s for 2 px and 30 ms);
- the reference ignores translation (mocap: 0.23 m/s median), which matters most at slow rotation.

## Conclusions

1. **The time unit isn't in a single pixel's event rate on this sensor.** Any per-pixel V estimator (1/Δt, leaky rate, bins) gives F in "per-event" units, no better than the current constant.
2. **It is in when the edge reaches neighbouring pixels.** The burst-onset time surface gives normal flow in px/s directly. As a relation, that would be "event timing → normal component of F", next to (or replacing) V, G → F.
3. **c cancels** in F = −V/|G| either way, so it can stay 1. The c values fitted here are in the frame's log units, which include the unknown APS response curve, and are not usable as the sensor's threshold.

## Open

- Time-surface parameters (burst gap, horizon, radius) are a speed-range trade-off; sweep on the simulator (M1) with exact ground truth.
- Low-speed bins need a reference that includes translation (mocap position) or the simulator.
- Stage B (inside the network) only makes sense once the network uses a timing-based normal flow.
