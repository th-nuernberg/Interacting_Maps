# Implementation architecture

Goal: one generic relaxation engine over a factor graph of **Maps** and **Relations**, so that new quantities (gyro, frame camera, depth, confidence, …) are added by declaration, never by changing the engine.

## Language and layers

| Layer | Language | Purpose |
| --- | --- | --- |
| `incomaps.reference` | Python + NumPy | Readable, synchronous reproduction of the paper. Ground truth for every other implementation. Never optimised. |
| `incomaps.core` | Python (Numba allowed) | Generic Map/Relation engine, config loader, synchronous solver. |
| `incomaps.sim` | Python | Analytic scene + trajectory generator, ideal DVS model, simulated gyro, ground truth. Adapters for ESIM / v2e output. |
| `incomaps.eval` | Python | Metrics, sweeps, plots. |
| `cpp/` (later, M4+) | C++20 + nanobind | Optimised and asynchronous engine with the same Map/Relation semantics. OpenEB delivers EVK4 events natively in C++; per-event Python dispatch cannot keep up at EVK4 rates. |

Rust would be a valid alternative to C++; the decision is C++ because of OpenEB.

## Core abstractions

**Map**
- `name`, `kind` ∈ {`field`, `global`, `constant`, `input`}
- `state`: ndarray (e.g. `(129, 129)` for I, `(128, 128, 2)` for F, `(3,)` for R)
- for `field` maps: a `Geometry` (resolution, intrinsics, pose, sensor id). Geometry lets relations link maps on different pixel grids (needed in WP2: EVK4 vs. frame camera).
- `constant` and `input` maps are never written by relations; `input` maps are written by sources (event binner, gyro integrator).

**Relation**
- `name`, the maps it connects, one `candidate(direction, maps) -> ndarray` per direction it updates, one rate `delta[direction]`.
- `residual(maps) -> ndarray` and `energy(maps) -> float` (for tests, convergence checks, later confidence weighting).
- Optional `weight(maps) -> ndarray` (per-pixel confidence, WP2; default 1).
- Update applied by the engine: `X ← (1 − δ·w) X + δ·w · candidate`.

**Network**
- Built from a YAML config (`configs/*.yaml`): maps, relations, rates, schedule, solver settings.
- Registry maps config `type:` strings to Map/Relation classes.

**Solver (synchronous, WP1)**
- Per time bin: write inputs → for `n_iter` iterations apply every (relation, direction) update in the configured order (Gauss–Seidel) or from a snapshot (Jacobi; selectable) → record diagnostics.
- Warm start: maps keep their state between bins (as in the paper).

**Scheduler (asynchronous, M5)**
- No thread per pixel. Each event marks the tile around its pixel as dirty.
- Work items are (relation, direction, tile) jobs; independent tiles run in parallel.
- Global maps (R, gyro bias) update from running least-squares sums (sufficient statistics updated per dirty tile), not full passes.
- The same job structure maps onto GPU batches of dirty tiles and later onto neuromorphic hardware.

## Required tests per relation

1. **Fixed point:** construct maps that satisfy the constraint exactly (from ground truth) → one update changes nothing (to float tolerance).
2. **Descent:** from a perturbed state, the relation's energy decreases for small δ.
3. **Gradient check:** where the candidate is a gradient step, compare with a finite-difference gradient of the energy.
4. **Convention check (R ↔ F, gyro ↔ R):** flow predicted from ground-truth R matches the simulator's ground-truth flow.

## Repository layout

```
Interacting_Maps/
  CLAUDE.md  README.md  TASKS.md  CMakeLists.txt
  configs/        network configs (YAML)
  docs/           specs (this folder)
  src/ include/   event-based C++ implementation (see cpp-implementation.md)
  tests/          Catch2 tests for the C++ code
  python/
    pyproject.toml
    src/incomaps/
      core/       map.py relation.py network.py solver.py registry.py config.py
      relations/  flow_constraint.py gradient.py rotation_flow.py gyro_rotation.py
      reference/  interacting_maps_ref.py   (standalone NumPy reference)
      sim/        scene.py trajectory.py camera.py dvs.py gyro.py groundtruth.py
      eval/       metrics.py sweep.py plots.py
    tests/
  experiments/    scripts + results (results not committed)
```
