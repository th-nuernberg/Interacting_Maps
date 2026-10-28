# Simulated gyro

In simulation the camera's true angular velocity is known at every instant. A simulated gyro is that ground truth plus a noise model, sampled on the same clock as the events.

## One trajectory drives everything

- A smooth rotation-only trajectory (e.g. sum of sinusoids or a spline on SO(3)) gives orientation q(t) and body angular velocity ω_true(t).
- The same trajectory renders the scene for the virtual DVS128 and feeds the gyro model, so events and gyro samples are aligned by construction.
- The simulator also outputs ground-truth R per bin (see below) and the true gyro bias.

## Gyro model (all knobs default to off except white noise)

```
ω_meas(t_k) = S · R_imu_cam · ω_true_cam(t_k + τ) + b(t_k) + n_k,   then quantise
b(t_{k+1})  = b(t_k) + σ_b · sqrt(Δt) · w_k        (bias random walk)
n_k ~ N(0, σ_n² / Δt)                                (white noise from noise density σ_n)
```

| Parameter | Meaning | Default |
| --- | --- | --- |
| `rate_hz` | sample rate | 1000 |
| `noise_density` σ_n | rad/s/√Hz, take from the datasheet of the IMU planned for hardware | datasheet |
| `bias_rw` σ_b | bias random walk | datasheet, or 0 |
| `bias_init` | initial bias | 0 |
| `scale_error` S | diagonal scale factor error | identity |
| `misalignment` | small rotation added to R_imu_cam | none |
| `time_offset` τ | gyro clock offset vs. event clock | 0 |
| `quantisation` | LSB in rad/s | none |
| `seed` | RNG seed | required |

Each knob must be switchable independently, so experiments can isolate its effect. ESIM can also generate IMU data from its trajectories; the built-in model is still needed for the analytic generator and for controlled knobs.

## Units and frames (most likely source of bugs)

- The paper's R is the **rotation per time bin**: F = m32(R × C).
- The gyro measures ω in **rad/s in the IMU frame**.
- **Sign.** A static point seen from a camera rotating with ω moves as Ṗ = −ω × P, so its viewing direction moves by (−Δθ) × C. With the paper's F = m32(R × C), R is therefore the **negative** of the camera's rotation over the bin.
- Gyro → R candidate for a bin [t₀, t₁]: `R_gyro = −R_cam_imu · ∫_{t₀}^{t₁} (ω_meas − b̂) dt`. Integrate trapezoidally over the samples inside the window **plus ω linearly interpolated at t₀ and t₁**. Integrating only between the first and last sample inside the window drops the two edge gaps, together up to one sample period (at 1 kHz and 5 ms bins: up to 20 % of the window).
- Ground-truth R per bin: minus the rotation vector of the camera's orientation change over the bin, expressed in the camera frame. For small rotations per bin this equals −∫ω dt; use the exact log map in the simulator and check the difference stays negligible.
- The sign of R × C must reproduce the simulator's ground-truth flow. Write a dedicated test: known ω → predicted F vs. projected pixel motion.

## New maps and relations

| Element | Kind | Content |
| --- | --- | --- |
| `omega` | input (global) | integrated gyro rotation for the current bin, raw (bias not removed) |
| `R_cam_imu` | constant | camera–IMU rotation (known in simulation; calibrated on hardware) |
| `gyro_bias` | global | slowly changing bias estimate b̂ |
| `GyroRotation` | relation | ω → R: candidate = −R_cam_imu (ω − b̂·T_bin); rate δ_ωR from gyro noise vs. visual R uncertainty |
| `GyroBias` | relation | R → b: candidate = ω/T_bin + R_imu_cam R/T_bin; very small rate δ_Rb (bias changes slowly) |

The gyro is a weak input: it nudges R, it never clamps it. With δ_ωR = 0 the network reduces exactly to the paper's.

**Bias observability.** b̂ is estimated from the disagreement between ω and R, while GyroRotation pulls R towards ω. If δ_ωR dominates the visual F → R pull (δ_FR), that disagreement collapses and b̂ cannot be estimated. Sweep δ_ωR and δ_FR jointly, not δ_ωR against δ_Rb alone.

## Experiments enabled

- Zero gyro noise, large δ_ωR → R effectively clamped; isolates the I/G/F part.
- Realistic noise → tests the actual fusion.
- Bias random walk on → tests whether b̂ tracks it.
- Time offset and misalignment → sensitivity to calibration errors.
- With vs. without gyro from random initialisation → does the β start-up ambiguity disappear?

## What simulation cannot settle

Real time offsets, temperature-driven bias drift and real DVS128 noise only appear on hardware. The DVS128 has no built-in IMU, so the first real-data check uses an external gyro, or the DAVIS rotation sequences of the Event Camera Dataset (Mueggler et al. 2017), which include IMU data.
