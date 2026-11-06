# Assumptions

State on 2026-11-06 (branch `rework/incomaps-alignment`). Every choice the current network rests on: physical assumptions, numbers, and implementation details that act like assumptions. Each one has a status, so that it is clear what is justified and what still has to be earned.

| Status | Meaning |
|---|---|
| **physics** | follows from the model (brightness constancy, projection, the sensor principle) |
| **measured** | supported by an experiment in this repository (link given) |
| **convention** | a choice that does not change results, only how they are written down |
| **inherited** | from the original code; no justification found |
| **open** | chosen deliberately, but the value still has to be determined (sweep, simulator M1) |
| **violated** | known to be false here; the effect is noted |

**Rule:** every inherited value is questioned and replaced by a grounded approximation where possible. The step 10, neutral 128 and bounds [0, 255] were chosen to map I directly onto 8-bit grey values; they were display choices, and the display now does that mapping itself.

Parameter names refer to `Parameters` in [include/interacting_maps.h](../include/interacting_maps.h). Reports are in [experiments/reports/](../experiments/reports/).

## 1. World and sensor

| # | Assumption | Status | Evidence / how to check |
|---|---|---|---|
| W1 | **Static scene, camera only rotates** (WP1). All pixels share one R. | **violated**, mildly | The mocap shows 0.23 m/s median translation in shapes_rotation. This matters most at slow rotation, where measured speeds are too high ([v-units](../experiments/reports/v-units.md)). Moving objects would also break it: the fit to R is plain least squares with no outlier handling. |
| W2 | **Brightness constancy**, V = −G·F: the scene's brightness pattern moves without changing (Lambertian surfaces, constant lighting). | physics | Holds for shapes_rotation. Frame gradient with gyro flow predicts polarity for 96 % of events ([polarity-check](../experiments/reports/polarity-check.md)); the remaining 4 % include noise and blur. |
| W3 | **An event means Δlog I = ±c, with the same c for ON and OFF.** | **violated**, measured | OFF : ON = 1.32 overall and 1.36 per pixel (IQR 1.22–1.53). The camera keeps viewing the same scene, so each pixel's log I returns to where it started. That implies c_ON ≈ 1.36 · c_OFF, and a pixel collects about 79 net OFF steps per minute. Without the decay, I drifts ([polarity-check](../experiments/reports/polarity-check.md)). Grounded replacement: separate ON and OFF steps, estimated per pixel. |
| W4 | **c never needs a value.** It cancels in F = −V G/\|G\|² as long as V and G use the same intensity unit; I is kept in "event steps". | physics | The c values fitted from frames include the unknown camera response and are not usable as the sensor's threshold ([v-units](../experiments/reports/v-units.md)). |
| W5 | **A pixel's own event rate does not carry the edge speed** on this sensor (bandwidth, refractory time). | measured | The burst rate grows as speed^0.24, where an ideal sensor gives speed^1 ([v-units](../experiments/reports/v-units.md)). Might not hold at slow speeds or in bright light; check on the simulator. |
| W6 | **The pixel delays an edge's onset by the same amount at neighbouring pixels**, so differences in arrival time carry the speed. | measured, indirectly | This is the basis of the timing mode. Normal speed is within 3–19 % of the truth above 240 px/s ([v-units](../experiments/reports/v-units.md)). Not tested directly (for example across contrast). |
| W7 | **Calibration:** `calib.txt` gives correct pinhole intrinsics and radial-tangential distortion (k1 = −0.37). | measured, indirectly | With distortion, \|ω\| ratio 1.00; pinhole only, 0.90–0.93 ([r-prototype](../experiments/reports/r-prototype.md)). The C++ `C`/`CCM` of the old pipeline is still pinhole only. |
| W8 | **The IMU frame equals the camera frame** (`R_cam_imu` = identity). | measured | Axes match the event-derived motion ([v-units](../experiments/reports/v-units.md)). Used only for fusion and plots. |
| W9 | **The IMU lags the events by 4 ms.** | measured | This lag maximises corr(event rate, \|ω\|) at 0.77 ([v-units](../experiments/reports/v-units.md)). Applied in plots and scoring only. |
| W10 | **The gyro is the truth for rotation** (bias and noise ignored). | open | It is the reference for every R score. A bias would show as a constant offset; not checked. |
| W11 | **Event timestamps are exact,** and the events are in time order. | open | The file is chronological. The C++ code stores times as `float` (see I1). |

## 2. Representation and conventions

| # | Assumption | Status | Notes |
|---|---|---|---|
| C1 | Camera frame x right, y down, z forward; F and G channel 0 = y (down), 1 = x; 3-vectors stored (y, x, z). | convention | Pinned by `[conventions]` tests. |
| C2 | R = −(camera angular velocity), as in the paper. | convention | Tested against explicit projection. |
| C3 | I, G and F on an H × W grid with central differences and replicate borders (the paper uses forward differences on (H+1) × (W+1)). | convention, small effect | Central differences skip the centre pixel, so G at a pixel does not see that pixel's own I. |
| C4 | The seed fixes the random initialisation and the update order. | convention | Tested (`[determinism]`). |

## 3. V and I (active in both modes)

| # | Assumption | Value | Status | Evidence / effect |
|---|---|---|---|---|
| V1 | **Every event adds a step of ±1 to I**, i.e. I is log intensity in units of c. | 1 (was 10, for display) | **physics** since 2026-11-06 | Exactly equivalent to the old 10: same R.txt, G = G_old/10 to float rounding. Still assumes the same c for ON and OFF (W3). |
| V2 | V → I adds the full step (`weight_IV`). | 1 | inherited ("weights 1" rule) | A different weight is equivalent to a different V1. |
| V3 | **I starts at, and decays back to, 0.** | 0 (was 128, for display) | **physics** since 2026-11-06 | The offset of log I is unobservable (the β ambiguity), so 0 is as good as any value. The display shows 0 as mid grey. |
| V4 | **I has no bounds.** | ±∞ (was [0, 255], for display) | **physics** since 2026-11-06 | The bounds were inactive with the decay (0.0 % of events). Without the decay, I now drifts freely instead of sticking at 0 (W3). |
| V5 | **I decays to 0** with time constant `decayParam`. So I is a *high-pass filtered* log I that forgets a static scene after about 0.1 s, not log I itself. | τ = 0.1 s | inherited value; the *need* is measured | It compensates the ON/OFF imbalance (W3): without it, polarity agreement falls from 84 % to 70 %. Grounded replacement: ON/OFF steps from W3, then a slow leak only for noise. |
| V6 | I ← G pushes I towards the integral of G, with full weight. | `weight_IG` = 1 | inherited ("weights 1") | Not examined separately. |

## 4. G

| # | Assumption | Value | Status | Evidence / effect |
|---|---|---|---|---|
| G1 | **G = ∇I** at the event pixel and its 4 neighbours after every event (G ← I with weight 1). | `weight_GI` = 1 | inherited ("weights 1") | G points the right way (median cos 0.92 with the frame gradient). It is wrong about 3.5× as often as the frames ([polarity-check](../experiments/reports/polarity-check.md)). Typical \|G\| corresponds to about one event step. |
| G2 | **F and G are not clamped** (`--clamp` turns the old ±255 / 1e-5 back on). | off (was on) | **physics** since 2026-11-06 | Was inactive in timing mode (the largest \|G\| was 68). In the old pipeline it bounded F = −V/G in flat regions; that now shows as large F where \|G\| ≈ 0. |
| G3 | **A pixel without events has V = 0**, an observation, not missing data (paper). Strictly, it means \|Δlog I\| < c. | — | physics, approximately | Replaced the G reset hack. Slow or low-contrast changes below c are treated as "no change". |
| G4 | **Silent relaxation:** G ← I at `rate_GI_silent`, and G loses its component along F at `rate_GF_silent`. | 22.3 /s each | inherited values | The former weight 0.2 per 10 ms, converted to a rate. Open (sweep). In timing mode, only F's direction is used. |
| G5 | **∇I and F stay constant over a pixel's silent time**, so the silent relaxation can be solved in closed form. | — | approximation | An event two pixels away can change ∇I in between. |
| G6 | G starts at 0. | — | convention | |

## 5. Old pipeline: F and R from V and G (default when `--rotationFromTiming` is off)

| # | Assumption | Value | Status | Notes |
|---|---|---|---|---|
| O1 | **F ← G, V:** F's component along G is set from V = ±10 (minimal-norm solution of V = −G·F). | `weight_FG` = 1 | physics, wrong input unit | F comes out in px per event, not px/s. Hence R has no unit (cos ≈ 0.02 against the gyro). |
| O2 | G ← F, V enforces the same constraint on G. | `weight_GF` = 1 | as O1 | Switched off in timing mode. |
| O3 | F ← R relaxes F towards m32(R × C) at `rate_FR`. | 644 /s | inherited | The former 4 passes of weight 0.8 per 10 ms. Lets about 10 F ↔ R trips happen per frame, where 100–300 are needed ([r-prototype](../experiments/reports/r-prototype.md)). |
| O4 | F → R least squares over **all** pixels, measured or not. | `weight_RF` = 1 | inherited | Dilutes the measurements ([r-prototype](../experiments/reports/r-prototype.md)). |
| O5 | Initial state: F uniform in [−1, 1], R uniform in [−10, 10]. | — | inherited | Timing mode starts from F = R = 0. |
| O6 | One event step per event, with relations in shuffled order; `lr` = 1. | — | inherited | |
| O7 | Gyro → R needs a scale (`imuScale`). | 1 | open, undefined | Meaningless while R has no unit. |

## 6. Timing pipeline (`--rotationFromTiming`)

| # | Assumption | Value | Status | Evidence / effect |
|---|---|---|---|---|
| T1 | **An edge's arrival time** at a pixel is set by short-term depression: O ← O + r(t − O), and r recovers with τ. | depression, τ = 20 ms, U = 1 | measured, best of those tested | Best for rotation, by a small margin ([onset-rules](../experiments/reports/onset-rules.md)). The rule can be swapped. |
| T2 | **Only same-polarity neighbours** belong to the same edge. | — | physics, approximately | An ON and an OFF edge close together are kept apart. A thin line (ON then OFF) gives two separate surfaces. |
| T3 | **The edge is locally straight and moves at constant speed** over a 5 × 5 neighbourhood and 30 ms (plane fit). | r = 2, horizon 30 ms | measured trade-off | The horizon caps the slowest speed at about r/horizon = 67 px/s, so slow speeds are overestimated ([v-units](../experiments/reports/v-units.md)). |
| T4 | A fit needs at least 4 recent neighbours; faster than 5000 px/s is rejected; the time residual has a floor of 0.1 ms. | 4, 5000 px/s, 0.1 ms | open | The floor limits how confident a fit can get. |
| T5 | **Measurement noise is Gaussian and independent between events**; its variance comes from the fit residual. | w = 1/σ² | open, probably violated | Neighbouring events share their onsets, so measurements are correlated and the network is overconfident. |
| T6 | **Only the normal component of F is measured** (aperture problem); the tangential part comes from R. | — | physics | |
| T7 | **A measurement fades exponentially**, and a new one replaces the old one at a pixel. | lifetime 10 ms, replace | measured, best of 10/30/100 ms | Old measurements describe a rotation that has already changed ([r-prototype](../experiments/reports/r-prototype.md)). Depends on how fast the motion changes. |
| T8 | **All measurements belong to one rigid rotation** (W1), fitted by plain least squares. | — | physics for WP1 | No outlier handling. Spikes in R (mean cos 0.94 vs median 0.99) may come from bad measurements. |
| T9 | **Unmeasured pixels follow R**; how strongly is set by the prior width. | σ = 27 px/s | measured, typical precision | κ = ∞ gives the same result ([r-prototype](../experiments/reports/r-prototype.md)). |
| T10 | M per pixel by numerical derivative through the lens model. | step 1e-6 rad | convention | Tested against the analytic pinhole result. |
| T11 | R is solved only when the evidence is well conditioned. | min eig > 1e-9 · max eig | convention | |
| T12 | The gyro, used as an input (`--fuseR`), is a measurement of R with σ. | 0.05 rad/s | open, a guess | Off by default. |

## 7. Implementation details that behave like assumptions

| # | Assumption | Status | Effect |
|---|---|---|---|
| I1 | **Times are `float`** in the C++ event path. | open, will break | Resolution is 4 µs at 30–60 s, but 61 µs at 500–1000 s. That approaches the 0.1 ms time floor (T4). Long recordings need `double`. |
| I2 | **`slice_events` includes both ends** of each 1 s slice. | bug | About 40 boundary events per minute are processed twice. Negligible here, but pinned by a test. |
| I3 | **Decay and relaxation are lazy:** a pixel is brought up to date only when an event reads it, or for display. | approximation, tested | Exact for the closed forms; uses the current R for the whole silent time (≈ 1.6 ms error window for F ← R). |
| I4 | Display and `R.txt` sample at 60 fps from relaxed copies, without changing the state. | convention | |

## 8. Evaluation

| # | Assumption | Value | Status | Notes |
|---|---|---|---|---|
| E1 | Frames give the true ∇log I near frame times (within 2 ms, motion-compensated, Gaussian σ = 1 px, \|∇\| ≥ 0.1). | — | open | Blur weakens the gradient as \|u\|^−0.2 ([v-units](../experiments/reports/v-units.md)). |
| E2 | R is scored only where \|ω_gyro\| > 1 rad/s, with 1 s rolling medians. | — | convention | Slow segments are excluded from the cos and ratio. |
| E3 | One dataset (shapes_rotation, 60 s), one seed. | — | open | Every number here is from this one recording. |

## Most consequential open assumptions

1. **W3 + V5: ON/OFF imbalance and the I leak.** They decide whether I and G can be trusted. G is currently the weaker map (84 % vs 96 %).
2. **W1: rotation only.** Translation biases slow speeds, and moving objects would corrupt R, since there is no robust fit (T8).
3. **V1, V3, V4, G2: the numbers 10, 128, 255** were display choices; replaced on 2026-11-06 by event units, offset 0, and no bounds (exactly equivalent in timing mode).
4. **T5: independent measurement noise.** It makes R overconfident and could explain the spikes.
5. **I1: float times.** Harmless now; a silent failure on long recordings.
