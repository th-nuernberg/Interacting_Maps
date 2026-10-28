# Work package 1 – original algorithm on simulated DVS128, then simulated gyro

Reproduce the paper's six-map network on a simulated DVS128 with full ground truth, make it fast and asynchronous, then add a simulated gyro that supplies R.

## Scope

- Simulation only. 128 × 128 pixels, rotation-only camera, static scene (the paper's world model).
- Input: events from an ideal DVS pixel model first, then v2e-style noise.
- Ground truth for every map at every bin.

## Steps (details and acceptance criteria in `TASKS.md`)

1. **Virtual DVS128.** Analytic generator: textured scene on a sphere or distant plane (rotation-only motion makes depth irrelevant), known smooth rotation trajectory, pinhole intrinsics, ideal log-intensity threshold-crossing events. Later: ESIM at DVS128 resolution; v2e noise.
2. **Reference implementation.** NumPy, synchronous, binned events as in the paper. Sweeps against ground truth fix the parameters the paper leaves out: bin length, every δ, iterations to convergence. Publishing these is already a contribution.
3. **Generic engine.** Same algorithm expressed through the Map/Relation core and `configs/base_network.yaml`; must match the reference.
4. **Optimised synchronous version.** C++ core with vectorised stencils (a multi-core CPU suffices at 128 × 128). Gate: matches the reference within tolerance and runs at least in real time.
5. **Asynchronous version.** Event-driven dirty-tile scheduler. Gate: same accuracy as step 4 at lower latency from event to updated R and F.
6. **Simulated gyro for R** (see `simulated-gyro.md`). Adds:
    - gyro rate ω as a weak input map,
    - camera–IMU rotation as a constant (known in simulation),
    - gyro bias b as a slowly changing global map, estimated from the persistent disagreement between ω and R,
    - relation ω → R (and R → b), rate δ set by gyro noise.
7. **IMU experiments.** Compare R error and convergence time with and without the gyro, across gyro noise, bias drift, time offset and misalignment.

The gyro relation (step 6) is part of the **reference** too: the reference must exist in two configurations, `base` (paper) and `base + gyro`.

## Hypothesis to test

A gyro fixes R absolutely, which pins F and therefore the scale β of G and I. The gyro should remove the start-up ambiguity (local optima from random initialisation) that the paper only avoids through warm starting. Experiment: random initialisation, rotation about the image centre, with and without gyro; count runs converging to the β-varying wrong solution.

## Parameters to determine (record final values here)

| Parameter | Value | How determined |
| --- | --- | --- |
| Bin length | open | sweep |
| δ_VF, δ_VG (flow constraint) | open | sweep |
| δ_IG, δ_GI | open | sweep |
| δ_RF, δ_FR | open | sweep |
| δ_ωR, δ_Rb | open | from gyro noise, then sweep |
| Iterations per bin | open | convergence curves |
| Gate: error tolerance vs. reference | open | propose after M2 |
| Gate: latency target | open | propose after M4 |
