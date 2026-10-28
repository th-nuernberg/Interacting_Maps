# Work package 2 – EVK4 + frame camera, full model

Build on the WP1 core and add maps in stages, each justified by an ablation, until the network extracts everything the two cameras and the IMU can jointly determine. **Do not start before WP1's asynchronous core has passed its latency gate** (sensor-setup work can run in parallel).

## Hardware

- Event camera: Prophesee EVK4 (1280 × 720), OpenEB / Metavision SDK.
- Frame camera: procured by an industrial partner to fit their processing hardware; model not yet known. Requirements: raw/linear output, manual exposure and gain, gamma off, hardware trigger or exposure/strobe output.
- IMU: to be chosen; its datasheet parameters feed the WP1 gyro model.

## Sensor setup

- **Time.** The EVK4's 4th-generation sensor has no trigger output, so the EVK4 listens: the frame camera's exposure or strobe output goes into the EVK4 trigger input. Rising and falling edges mark each exposure window in the event clock and correct drift every frame. With an IMU as well, a microcontroller triggers all sensors and its pulse is logged by the EVK4. Note: external trigger events must be explicitly enabled to appear in recordings.
- **Space.** Preferred: coaxial setup with a beam splitter, so alignment is a depth-independent homography. Side by side, near objects are misaligned unless depth D is in the network. Calibrate the EVK4 with a blinking checkerboard on a screen, then both cameras' relative pose jointly.
- **Photometry.** DVS events signal log-intensity changes, so V is a log-intensity derivative. The frame observation is log of linear raw intensity, with auto exposure, gain and gamma disabled.

## Map additions, in order

1. **Frame observation** I_frame → I: I ← (1 − δ_obs) I + δ_obs I_frame, weighted per pixel (low near saturation, low with motion blur). Absolute contrast from frames fixes the scale and sign of I and G, and with them F and R (breaks the β ambiguity).
2. **Confidence maps** for every state map; δ scaled by source confidence (reliability weighting).
3. **Gated relations:** robust losses that switch a relation off where its residual stays large (occlusions, independent motion, flicker).
4. **Full IMU:** gyro → R as in WP1, accelerometer → T for metric scale.
5. **Depth D and translation T** via the split F = S + P (S from rotation, P parallax from translation and depth).
6. **Ownership map** with several R/T pairs for independently moving objects.
7. **Temporal prediction** (previous state propagated by F, as a relation) and the **shading/reflectance split** of I.

## Scale

Each retinotopic map lives on its own sensor's grid; relations resample between grids. The EVK4 has about 56 × the pixels of the DVS128. All updates except the F → R least-squares fit are local, so GPU batches of dirty tiles are the natural route. Coarse-to-fine pyramid maps handle large flows.

## Evaluation per stage

Ablation against the previous stage, observability analysis of the new maps, and a failure map from simulation sweeps (see `roadmap-and-principles.md`).
