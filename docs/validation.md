# Validation

Test on a ladder from exact synthetic ground truth to real hardware, and report the metrics the paper left out.

| Environment | Ground truth available | Used for |
| --- | --- | --- |
| Own analytic generator (`incomaps.sim`) | Exact I, G, V, F, R, gyro bias | WP1 unit and regression tests |
| [ESIM](https://rpg.ifi.uzh.ch/esim.html) | Events, frames, IMU; pose, velocity, depth, optical flow | WP1 virtual DVS128; WP2 fusion and translation |
| [v2e](https://github.com/SensorsINI/v2e) | Realistic noise, threshold mismatch, low light | Robustness in both work packages |
| CARLA ([eCARLA-scenes](https://arxiv.org/abs/2412.09209)) | Events, grayscale frames, optical flow | WP2 driving scenes |
| Event Camera Dataset, rotation sequences (Mueggler et al. 2017) | IMU and motion-capture pose | First real-data check of the rotation-only model |
| Own rig on a pan-tilt head with gyro | R from gyro and encoders | WP2 on real hardware |

Real-world ground-truth flow is hard to obtain; some datasets use only purely rotational motion measured with an IMU ([survey](https://arxiv.org/abs/1904.08405)). That matches WP1's rotation-only model.

## Metrics

- **F:** average endpoint error (AEE) and average angular error (AAE), over pixels with |G| above a threshold and over all pixels (report both).
- **R:** angle and norm error of the rotation per bin vs. ground truth.
- **I:** error after removing the scale and offset the network cannot determine (fit I_gt ≈ a·I + c, report residual RMS and the fitted a; a < 0 means the sign ambiguity was resolved wrongly).
- **G:** same scale-corrected comparison as I.
- **Gyro bias:** tracking error of b̂ vs. true bias.
- **Dynamics:** iterations / events to convergence; latency from event to updated R and F (asynchronous versions).
- **Robustness:** failure maps over texture density, contrast, angular speed, noise level and gyro knobs.

## Experiment hygiene

- Every run writes its config, seed, git hash and metrics to `experiments/results/<run-id>/`.
- Plots are generated from saved metrics, never from in-memory state.
