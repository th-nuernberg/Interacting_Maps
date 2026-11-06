# Tasks – WP1

Work top to bottom. Each milestone ends with its acceptance criteria passing in `pytest` (or a saved experiment result where noted). Tick items as you go and record decided parameter values in `docs/wp1-spec.md`.

## M0 – Rework of the existing C++ implementation

Fix implementational flaws in `src/` one commit at a time, each with a Catch2 regression test in `tests/`. Behaviour that deviates from the paper on purpose (V injected into I with decay, clamping, central differences) is kept as the default but made switchable and documented in `docs/cpp-implementation.md`.

- [x] Build from a fresh clone; Catch2 tests wired into CTest.
- [x] Results folder creation must not edit `.gitignore`.
- [x] `update_GF`: y component computed from the x component.
- [x] Main loop reads `intervals[i+1]` past the end.
- [x] `switch` over update order falls through from `default`.
- [x] `Vector2Tensor` uses `cols()` (= 1) instead of `size()`.
- [x] `computeGradient` border cases: last row/column read the wrong cells (replicate-border intent kept).
- [x] G→I residual (`GIDiff`) random-initialised and only refreshed at the event pixel.
- [x] F→R least squares: right-hand side `B` only contains pixels that saw an event, and goes stale after global F←R updates.
- [x] `read_single_line_txt` parses floats with `std::stoi` (focal lengths truncated, distortion lost).
- [x] Calibration: fx/fy swapped in view angles; principal point ignored in C. (Lens distortion is still not modelled; events are used as recorded.)
- [x] Camera frame is left-handed (y up, z forward) while cross products are right-handed; switch to x right, y down, z forward.
- [x] Seeded, reproducible runs (`--seed`, Eigen `setRandom`).
- [x] Duplicate short CLI flags (`-f`, `-s`, `-b`).
- [x] Paper deviations switchable: decay, clamping, per-frame G reset under `IMAGES`.
- [x] IMU fusion: gyro (x, y, z) was written into R's (y, x, z) slots unrotated and with the wrong sign; now rotated by `R_cam_imu`, reordered and negated.
- [x] Lazy decay: gradients mixed decayed and stale potentials (only the event pixel was decayed); now the whole stencil (Manhattan distance 2) is decayed, and output frames show I decayed to the frame time.
- [x] Pixels without events never updated (G stale wherever an edge passed; per-frame G reset was a workaround): G ← I and G ← F with V = 0 integrated lazily over each pixel's silent time; G reset off by default.
- [ ] Silent-time rates `rate_GI_silent`, `rate_GF_silent`: defaults reproduce one paper update per time step; determine by sweep (M2).
- [x] Event file loaded once into memory (was re-read from the start for every interval); RTF reported per run and per interval.
- [x] Event-driven F ← R (closed-form B relaxation, lazy F); display decoupled from `timeStep` (`--fps`, default 60). RTF on shapes_rotation: 0.51 → 0.11.
- [x] All relation weights default to 1 (no undocumented damping); continuous-time rates decoupled from the weights and kept as open parameters.
- [ ] Continuous-time rates `rate_GI_silent`, `rate_GF_silent`, `rate_FR`: no neutral value exists; determine by sweep (M2).
- [x] (with `--rotationFromTiming`) R does not track the gyro (|corr| < 0.25, synchronous and event-driven). Diagnose against the NumPy reference / simulator ground truth.
- [x] V-unit experiment, stage A (`experiments/v_units.py`, report `experiments/reports/v-units.md`): per-pixel event rates carry almost no speed on shapes_rotation; burst-onset time surfaces give normal flow in px/s.
- [x] Offline prototype (`experiments/r_prototype.py`, report `experiments/reports/r-prototype.md`): standing timing-based normal-flow constraints give ω in rad/s (median cos 0.99 with the gyro, |ω| ratio 1.00, τ = 10 ms); the F ↔ R loop converges to the same steady state but needs ~100–300 trips per 1/60 s.
- [x] Onset rules compared (`experiments/onset_rules.py`, report `experiments/reports/onset-rules.md`): soft onset by depression (τ 10–20 ms) slightly best for rotation, hard gap 10 ms best for single normal flows; both swappable via `experiments/event_maps.py`.
- [x] Rotation from event timing in C++ (`--rotationFromTiming`, `include/timing.h`): reproduces the prototype (median cos 0.992, |R|/|ω| 0.995, RTF 0.26); gyro as a weak measurement of R; `read_imu` axis order fixed.
- [x] Timing mode is the default (2026-11-06); the old pipeline runs with `--legacy true`. Still to validate on the simulator (M1).
- [ ] Couple G ↔ F in real units (V = −G·F in log units/s); a bound from silence was falsified at edge fronts (`experiments/reports/silent-bound.md`).
- [ ] Decide how event timing enters the network (time-surface normal flow as a relation) and sweep its parameters on the simulator (M1).
- [ ] IMU fusion units: choose `imuScale` (rad/s → R units). Open: R has no physical time unit yet, see `docs/cpp-implementation.md`. Needs V per time bin (paper) or a calibrated event contribution.

## M1 – Simulator with ground truth (`python/src/incomaps/sim`)

- [ ] `trajectory.py`: smooth rotation-only trajectory (sum of sinusoids per axis, configurable amplitude/frequency), returns orientation and body ω at arbitrary t.
- [ ] `camera.py`: pinhole model for 128 × 128, intrinsics from config; viewing directions C; projection m32 and lifting m23 with documented sign conventions.
- [ ] `scene.py`: textured scene on a sphere around the camera (procedural texture, e.g. band-limited noise plus edges; contrast and texture density configurable). Renders log intensity for any orientation, sub-pixel accurate.
- [ ] `dvs.py`: ideal DVS pixel (log-intensity threshold crossings, per-pixel reference level, contrast threshold configurable), adaptive time stepping so no crossing is missed. Event binner → V per bin.
- [ ] `groundtruth.py`: per bin I (129 × 129), G, V, F (projected pixel motion), R (rotation vector of the orientation change over the bin, camera frame).
- [ ] `gyro.py`: model from `docs/simulated-gyro.md`, every knob independent, seeded.
- **Accept:** (a) ground-truth F equals m32(R_gt × C) within tolerance (convention test); (b) summed events per pixel ≈ ΔlogI / threshold; (c) gyro with all knobs off integrates to R_gt per bin; (d) identical seeds → identical outputs.

## M2 – NumPy reference (`incomaps.reference`)

- [ ] One standalone file implementing the paper exactly (`docs/paper-summary.md`), maps V, I, G, F, C, R; all six update directions; warm start between bins; random init with seed.
- [ ] Optional gyro configuration: maps ω, R_cam_imu, gyro_bias; relations GyroRotation and GyroBias (`docs/simulated-gyro.md`). δ_ωR = 0 must reproduce the base algorithm exactly.
- [ ] Per-relation tests: fixed point, descent, finite-difference gradient (see `docs/design.md`).
- [ ] `incomaps.eval.metrics`: all metrics from `docs/validation.md`.
- [ ] Sweep script for bin length, δs, iterations → convergence curves and error tables saved to `experiments/results/`.
- **Accept:** on the analytic scene with moderate rotation, warm-started reference converges to scale-corrected I/G and F/R close to ground truth (propose numeric thresholds from the sweep and ask before fixing them).

## M3 – Generic engine (`incomaps.core`)

- [ ] Map, Geometry, Relation base classes; registry; YAML config loader; synchronous solver (Gauss–Seidel and Jacobi).
- [ ] Relations as classes: FlowConstraint, Gradient, RotationFlow, GyroRotation, GyroBias.
- [ ] `configs/base_network.yaml` and `configs/gyro_network.yaml` load and run.
- **Accept:** engine output matches the reference to float tolerance for both configs on the same seed and schedule; adding a dummy relation via config requires no change in `core/`.

## M4 – Optimised synchronous C++ core

- [ ] `cpp/` with CMake, C++20, nanobind module exposing the same Map/Relation semantics; vectorised stencils, OpenMP over rows.
- **Accept:** matches the reference within the agreed tolerance; ≥ real time on the analytic scene (bins processed faster than they are generated).

## M5 – Asynchronous version

- [ ] Event-driven dirty-tile scheduler; global maps from running least-squares sums.
- **Accept:** same accuracy as M4; lower median latency from event to updated R and F.

## M6 – Gyro experiments (results, not tests)

- [ ] R error and convergence time with vs. without gyro, over noise density, bias random walk, time offset, misalignment.
- [ ] Start-up ambiguity: random init, rotation about the image centre, N seeds, with vs. without gyro; fraction converging to the wrong (β-varying) solution.
- [ ] Short report in `experiments/reports/gyro.md`.

## Later

- ESIM and v2e adapters; Event Camera Dataset rotation sequences as first real data.
- WP2 per `docs/wp2-spec.md`.
