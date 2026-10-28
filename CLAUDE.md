# INCOMAPS – context for Claude Code

INCOMAPS restarts and extends the **Interacting Maps** algorithm (Cook, Gugelmann, Jug, Krautz, Steger, "Interacting Maps for Fast Visual Interpretation", IJCNN 2011). A small network of recurrently coupled, retinotopic maps (intensity, gradient, optical flow, camera rotation, …) relaxes from event-camera input to a mutually consistent scene interpretation. Each map is pulled towards agreement by soft constraints ("relations") with the maps it touches.

Read these before writing code:

| File | What it holds |
| --- | --- |
| `docs/paper-summary.md` | The original algorithm: maps, relations, update rules, known limitations |
| `docs/design.md` | Implementation architecture: Map/Relation core, scheduler, language choices |
| `docs/wp1-spec.md` | **Current work package**: reference implementation on simulated DVS128, then simulated gyro |
| `docs/simulated-gyro.md` | Gyro model, unit/frame conventions, gyro → R relation |
| `docs/validation.md` | Test environments, ground truth, metrics |
| `docs/wp2-spec.md` | Later work package: EVK4 + frame camera, uncertainty maps, sync and alignment |
| `docs/roadmap-and-principles.md` | Expansion ideas, multisensory principles, limits of the setup |
| `TASKS.md` | Ordered task list with acceptance criteria — work through it top to bottom |
| `docs/cpp-implementation.md` | The existing event-based C++ implementation: how it maps onto the paper, deviations, known issues |

## Where we are

- This repository holds an existing **event-based C++ implementation** (`src/`, `include/`, CMake, Catch2 tests in `tests/`). It is being reworked under milestone M0 in `TASKS.md`: known implementational flaws are fixed one commit at a time, each with a regression test.
- The Python side (`python/`, package `incomaps`) is still a scaffold. After M0, continue with M1.
- Hardware decided: Prophesee **EVK4** event camera (for WP2). The frame camera for WP2 is procured by an industrial partner to fit their processing hardware; model not yet known.
- WP1 is **simulation only**: virtual DVS128 (128 × 128), rotation-only camera, static scene, simulated gyro.

## Non-negotiable design rules

1. **Reference first.** A small, readable NumPy implementation of the original synchronous algorithm is the ground truth for all later optimised / asynchronous / C++ versions. Never optimise the reference; keep it obviously correct.
2. **Generic core.** Two abstractions only: `Map` and `Relation`. The network is declared in a config file (`configs/*.yaml`). Adding a quantity = new Map/Relation class + config lines. The scheduler/solver must never contain map-specific code.
3. **Every relation is tested in isolation:** (a) its fixed point holds when the constraint is satisfied; (b) its residual decreases monotonically for small δ; (c) the analytic gradient matches a finite-difference gradient.
4. **Ground truth everywhere.** The simulator returns ground truth for every map (I, G, V, F, R, and later gyro bias). Report per-map errors, not just final flow.
5. **Weak inputs.** Inputs (V from events, ω from the gyro, later frames) only *nudge* maps with their own δ; they are never clamped.
6. **Determinism.** All simulations take an explicit seed; results must be reproducible bit-for-bit on the same machine.

## Conventions (keep consistent — most bugs will be here)

- Image coordinates: `x` = column (right), `y` = row (down), arrays indexed `[y, x]`. Map shapes follow the paper: V, G, F, C are 128 × 128; I is 129 × 129 (forward differences).
- Intensity is **log intensity**. V is the temporal derivative of log intensity per time bin.
- Camera frame: right-handed, z along the optical axis, x right, y down. C[y, x] is the unit (or z = 1) viewing direction of the pixel.
- R is the **rotation vector per time bin** (radians per bin, camera frame), so that F = m32(R × C). The gyro measures ω in rad/s in the IMU frame; the gyro relation integrates ω over the bin window and rotates it into the camera frame. Write the sign convention down in code next to `m32`/`m23` and test it against the simulator's ground-truth flow.
- δ naming: `delta_AB` = rate for the update of B driven by A (e.g. `delta_IG`: I → G).

## Stack

- Python ≥ 3.11, NumPy, SciPy, Matplotlib, PyYAML, pytest. Numba allowed for the optimised synchronous Python path. Lives in `python/` (`cd python && pip install -e ".[dev]" && pytest`).
- C++20 with Eigen (Tensor module), OpenCV, Boost.program_options, HDF5 C++, autodiff; dependencies via CPM where possible. Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build`.
- The existing C++ code is event-based and predates the reference. Until the NumPy reference exists, C++ fixes are checked by unit tests against hand-derived values. Generic Map/Relation C++ core with nanobind bindings is milestone M4; do not start it before the reference passes its tests.
- Later: OpenEB / Metavision SDK for the EVK4 (WP2 only).

## Open questions (do not invent answers — make them parameters and flag them)

- The paper does not report bin length, any δ value, or iterations to convergence. Determine them by sweeps against ground truth (M2) and record the chosen values in `docs/wp1-spec.md`.
- Numeric gate thresholds (error tolerance, latency target) are not yet set. Propose values from the sweep results and ask.
- Frame camera model for WP2 is not yet known.
