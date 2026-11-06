# R now instead of R late; F → G in sign (shapes_rotation)

2026-11-06 · C++: `--predictRotation`, `--signCorrectG` (both on by default since this report) · scoring: `experiments/rotation_accuracy.py`, `experiments/polarity_check.py` · results: `experiments/results/rotation_accuracy/`, `experiments/results/sign_correction_frames/`

## Changes

**1. Delay correction (`RotationFromFlow::rotationAt`).** R was 12 ms late ([polarity-sign](polarity-sign.md)). Each measurement describes the motion at the time its edge passed, not now:
- **Dating:** a normal flow is dated at the mean of the onsets its plane was fitted to (`NormalFlow::time`). The gyro, when fused, is dated at its timestamp minus the 4 ms IMU lag.
- **Model:** R(tᵢ) = R(t) + Ṙ·(tᵢ − t) for every measurement i, fitted by least squares over [R(t), Ṙ]. That needs the time moments Σ tᵢᵏ MᵀΛM (k = 0, 1, 2) and Σ tᵢᵏ Mᵀη (k = 0, 1), which fade as a whole like the existing sums. Times are kept relative to an epoch that is rebased every 10 s.
- **Cost:** the 6×6 system is solved at most once per ms of data time, and R is extrapolated with Ṙ in between (a computational choice, `rotationSolveInterval`). If the evidence spans too little time for a rate, R falls back to the average.

**2. F → G in sign (`correctGradientSign`).** The polarity is the sign of V = −G·F, so it says which way G points along F. G's component along F, uncertain with σ_G = 0.5 steps/px, is moved to its posterior mean given a polarity that is right with probability ρ = 0.96 (frame-gradient agreement):

  g ← g + s·σ_G·φ(z)·(2ρ − 1) / (ρΦ(z) + (1 − ρ)(1 − Φ(z))),  s = −polarity, z = s·g/σ_G

- A weak contradiction is overruled. A strong contradiction is left alone, since the polarity is then likely wrong.
- The correction is kept as a map δG on top of ∇I, which forgets with I's τ = 0.06 s (it describes the same intensity pattern). It is applied after the event. So it reaches later events of the same edge, and their G → F check, without scoring the event with its own polarity.

## Results: 60 s, scored against the IMU (|ω| > 1 rad/s, after 2 s)

| run | cos median / mean | frames cos < 0.9 | … at > 100 rad/s² | \|ω\| ratio | \|Δω\| rms (rad/s) | per-axis corr | delay |
|---|---|---|---|---|---|---|---|
| default before | 0.9918 / 0.9399 | 11.97 % | 24.6 % | 1.000 [0.87, 1.15] | 1.556 | 0.958 / 0.964 / 0.956 | 12 ms |
| delay correction | 0.9975 / 0.9860 | 2.06 % | 5.1 % | 1.102 [1.05, 1.18] | 0.973 | 0.989 / 0.991 / 0.988 | 4 ms |
| sign correction of G | 0.9918 / 0.9398 | 12.00 % | 24.7 % | 1.000 [0.87, 1.15] | 1.557 | 0.958 / 0.964 / 0.956 | 12 ms |
| **both (new default)** | **0.9975 / 0.9860** | **2.06 %** | **5.1 %** | 1.103 [1.05, 1.18] | **0.975** | **0.989 / 0.991 / 0.988** | **4 ms** |
| both, lifetime 5 ms | 0.9979 / 0.9886 | 1.72 % | 4.0 % | 1.114 [1.06, 1.18] | 0.948 | 0.992 / 0.993 / 0.990 | 3 ms |
| both, lifetime 20 ms | 0.9958 / 0.9727 | 4.80 % | 12.3 % | 1.106 [1.02, 1.21] | 1.212 | 0.978 / 0.983 / 0.981 | 7 ms |

**With the IMU fused into R** (`--fuseR`, σ = 0.05 rad/s). These rows are scored against the same gyro they used, so they are optimistic:

| run | cos median / mean | frames cos < 0.9 | \|ω\| ratio | \|Δω\| rms | per-axis corr | delay |
|---|---|---|---|---|---|---|
| default before, IMU fused | 0.9920 / 0.9428 | 11.22 % | 1.003 | 1.482 | 0.958 / 0.965 / 0.962 | 13 ms |
| both, IMU fused | 0.9977 / 0.9886 | 1.66 % | 1.096 | 0.883 | 0.989 / 0.991 / 0.992 | 4 ms |

**G** (per-event dumps, every 20th event):

| | polarity agreement (next event, gyro flow) | G against the frame gradient: points the right way / median cos | polarity agreement on events near frames (frames: 0.959) |
|---|---|---|---|
| before | 0.850 | 92.0 % / 0.934 | 0.869 |
| sign correction | **0.872** | **93.2 %** / 0.932 | **0.890** |

Run time: RTF 0.21 with both changes, 0.29 before. Solving at most once per ms is cheaper than solving the 3×3 system at every event.

## Findings

1. **The delay correction removes most of R's error.** Mean cos goes from 0.940 to 0.986, frames below 0.9 from 12 % to 2 %, the error from 1.56 to 0.98 rad/s rms, and the delay from 12 to 4 ms. The spikes at fast changes drop by a factor of 3–10 (`spikes_by_acceleration.png`). In the axis plot the corrected R follows the gyro without a visible lag (`omega_window.png`).
2. **It exposes a magnitude bias: |ω| is now 10 % too large.** The lag had hidden it: a late estimate averages directions that differ, which shrinks it, and that happened to cancel the overestimate of the normal-flow speeds (consistent measurements read 1.14× too fast; [polarity-sign](polarity-sign.md)). This is the next error to fix, at its source (the plane fit's speed bias, [v-units](v-units.md)), not by scaling.
3. **With the delay modelled, the lifetime trade-off moves.** 5 ms is now better than 10 ms on every score except the ratio; 20 ms is worse. The lifetime stays an open parameter (simulator sweep).
4. **The sign correction improves G, confirmed by an independent reference.** It points the right way more often (frame gradient: 92.0 → 93.2 %) and predicts the next event's polarity better (0.850 → 0.872). It doesn't change R, as expected for one rigid rotation (G only weights the evidence; see [polarity-sign](polarity-sign.md)).
5. **Fusing the IMU adds little on top of the events** (mean cos 0.986 → 0.989), because about 10 000 event measurements outweigh one gyro sample with σ = 0.05 rad/s. The gyro's weight is the open σ.
6. **A remaining glitch:** at 25.55 s both versions spike for one or two frames. It is not caused by the delay correction; it marks a short failure of the measurements, still to be examined.

## Remaining

- The speed bias of the normal flow (ratio 1.10).
- The 4 ms left of the delay (part may be the IMU-lag estimate itself).
- The single-frame glitches.
- The evidence lifetime (5 ms looks better).
