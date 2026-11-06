# Tests: what is checked, and how

State on 2026-11-06 (branch `rework/incomaps-alignment`). The plan for future validation (simulators, datasets, metrics) is in [validation.md](validation.md). This page lists what exists now.

There are three layers:

| Layer | Checks | Ground truth | Where |
|---|---|---|---|
| **Unit tests** (56 Catch2 cases, 633 assertions) | each relation, helper and default does what it should | analytic: hand-built states, limits, equivalences | `tests/` |
| **Method experiments** (Python) | does an idea work on real data before it is ported? | gyro, DAVIS frames | `experiments/`, reports in `experiments/reports/` |
| **End-to-end runs** (C++ binary) | does the whole network track the camera, how fast, and did a change break anything? | gyro (display/scoring only), earlier outputs | `output/<run>/`, `experiments/plot_r_vs_gyro.py` |

## How to run

```bash
cmake --build build && ctest --test-dir build --output-on-failure   # all unit tests
build/tests/im_tests "[timing]"                                       # one tag
build/tests/im_tests --list-tags
```

`tests/tests.cpp` is the old hand-rolled harness for the xtensor API. It does not compile on this branch and is not built.

## 1. Unit tests

Each test checks one relation or helper against something known independently: an analytic result, a limit, or an equivalent but slower computation. Tags in brackets select groups.

### Conventions and calibration `[conventions]` `[calibration]`

| Test | Guarantees |
|---|---|
| crossProduct3x3 matches the right-handed cross product in (y, x, z) storage | the storage order of R does not flip the cross product |
| F = m32(R × C) with R = −(camera rotation) reproduces projected pixel motion | the sign and axis convention of R against a projected 3-D rotation |
| update_RIMU turns a gyro reading into the R that reproduces the projected flow | the gyro → R path uses the same convention |
| find_C uses the principal point and per-axis focal lengths | the camera rays C |
| get_calibration_data pairs the vertical view angle with fy | no x/y mix-up in the calibration |

### Input/output `[io]`

| Test | Guarantees |
|---|---|
| create_results_folder creates output/&lt;name&gt; without touching .gitignore | runs don't edit repository files |
| read_single_line_txt keeps fractional and negative values | settings are parsed exactly |
| splitTimeInterval returns slice boundaries including both ends | no events lost at slice edges |
| load_events_txt skips a header and slice_events keeps read_events' inclusive bounds | the in-memory loader equals the old file-based reader |
| read_imu keeps the file's (x, y, z) axis order | regression for the fixed axis swap |

### Core maps and relations `[update]` `[flow]` `[gradient]` `[event_step]`

| Test | Guarantees |
|---|---|
| update_GF / update_FG blend each component towards its own candidate | no component mixing in the G ↔ F relations |
| computeGradient uses replicate borders on all four sides | border handling of ∇I |
| computeGradient on a 3-tensor differentiates channel 0 along y and channel 1 along x | channel order of the gradient |
| G-I residual divergence vanishes when G equals the gradient of I | the G ← I relation has the right fixed point |
| event_step moves G at the event pixel towards the current gradient of I | an event actually updates G |
| event_step ignores unknown update ids instead of running update_FG | no silent fallback to a wrong relation |
| Vector2Tensor keeps every element of the vector | conversion helper |

### Lazy decay and silent relaxation `[decay]` `[silent]`

These check the closed-form "catch up on the time since the last event" rules against the step-by-step updates they replace.

| Test | Guarantees |
|---|---|
| decayNeighbourhood brings every pixel read by event_step's gradients to the event time | no stale neighbour enters a gradient |
| decayedCopy decays every pixel and leaves the state untouched | display reads don't change the network |
| relaxSilentG without G ← I equals repeated V = 0 update_GF steps | closed form = iterated steps (one relation) |
| relaxSilentG without G ← F equals repeated update_GI steps | closed form = iterated steps (other relation) |
| relaxSilentG: V = 0 removes only G's component along F | silence means "no brightness change along F", nothing more |
| relaxSilentG: stale G at a pixel an edge has left goes to zero | replaces the G reset hack with a justified rule |

### Event-driven F ← R `[async]` `[rotation]`

| Test | Guarantees |
|---|---|
| m32_pixel equals m32 for R × C | per-pixel version of the projection |
| Event-driven F ← R over one time step equals the synchronous passes | asynchronous = synchronous result |
| relaxB matches the running sum of every pixel relaxed with relaxFR | the O(1) running sum B stays exact |
| register_R_point brings B up to date with a changed F | B stays exact after a single-pixel change |
| F → R running sum covers every pixel's current flow after initialisation | B is complete at start |
| F → R recovers R from a flow field generated by R after a global F ← R pass | the R ← F relation inverts F ← R |

### Rotation from event timing `[timing]`

The pipeline per event: onset map O → normal flow N → flow evidence (Λ, η) → R, and the stored F read from it.

| Test | Guarantees |
|---|---|
| Hard-gap onset keeps the first event of a burst and restarts after a gap | `gap` rule |
| Depression onset lies between latest event (τ → 0) and first onset (τ → ∞) | `depression` rule has the right limits |
| Depression onset resets after a long silence, also when an event keeps resource (U &lt; 1) | regression for the U·r bug |
| makeOnsetRule builds the named rules and rejects unknown names | rules are swappable by name |
| fitNormalFlow recovers direction and speed of a plane wave of arrival times | N in px/s from O |
| fitNormalFlow needs four recent neighbours and reports more variance for noisy times | N refuses weak fits and reports its uncertainty |
| Rotation model without distortion equals F = m32(R × C) of the network | the new motion model M agrees with the old relation |
| Barrel distortion slows the predicted motion towards the border, not at the centre | M includes the lens (k1 = −0.37) |
| Running sums equal the faded evidence of every pixel (replace and accumulate) | H, b stay exact under fading, both evidence modes |
| RotationFromFlow recovers R from normal-flow measurements generated by R | R ← evidence inverts M |
| R is solved only when new evidence arrived and R is read | "unchanged R is not recalculated" |
| Gyro-like measurements of R alone determine R | the `--fuseR` input path |
| readFlow gives R's prediction without evidence and follows strong evidence along n | the stored F: prior where unmeasured, data along n where measured |

### Defaults, deviations and determinism `[deviations]` `[determinism]`

These pin the decisions so they can't change unnoticed.

| Test | Guarantees |
|---|---|
| All relation weights default to 1, inputs stay off | "weights 1 until we know why not" |
| G reset is off and silent relaxation on by default | the hack stays off |
| Event-driven F ← R is on by default | |
| clampMaps switches the F/G clamping off | inputs are never clamped |
| initOpticalFlowState is reproducible for a given seed | determinism |

### Contrast steps `[contrast]`

| Test | Guarantees |
|---|---|
| Equal contrast steps are ±1, a fixed ratio scales only ON events | `equal`, `fixed` |
| Global contrast ratio is the OFF/ON count ratio, used causally | `global` uses counts before the event |
| Per-pixel contrast ratio moves from the global ratio to the pixel's own counts | `pixel` shrinkage |
| Unknown contrast step modes are rejected | |
| I is in event steps without bounds, and F and G are not clamped by default | the display-derived values are gone |

### Display `[plot]` `[imu]`

| Test | Guarantees |
|---|---|
| GyroReference interpolates the gyro and rotates it into the camera frame | the reference curve in the plot |
| RotationPlot's rolling statistics are 1 for a perfect estimate and skip slow reference motion | the cos and ratio strips |
| RotationPlot renders an image of the requested size | layout does not break the frame |

## 2. Method experiments (Python, real data)

Each answers one question on `shapes_rotation` before anything is ported. References are external to the network: the gyro (4 ms lag corrected) and the DAVIS frames.

| Experiment | Question | Result |
|---|---|---|
| [v-units.md](../experiments/reports/v-units.md) | Which reading of the events gives V (and so F, R) in real units? | Per-pixel event rates carry almost no speed (rate ∝ speed^0.24). Arrival times across neighbours (plane fit) give normal flow in px/s, within 3–19 % above 240 px/s. |
| [r-prototype.md](../experiments/reports/r-prototype.md) | Does the F ↔ R loop with fading measurements settle into a state that tracks the rotation? | Yes: steady state cos 0.99 (median), ratio 1.00, no fitted scale. The loop reaches the same fixed point but needs 100–300 trips per frame; solved directly instead. Lens distortion is needed for the magnitude (pinhole: −7 to −10 %). |
| [onset-rules.md](../experiments/reports/onset-rules.md) | Which rule turns a pixel's burst into an arrival time? | Any burst rule beats the latest event. Depression τ = 10–20 ms, U = 1 is best for rotation (mean cos 0.938); hard gap is best for single measurements. |
| [polarity-check.md](../experiments/reports/polarity-check.md) | Is the network's G good enough to predict event polarity, sign(−G·u)? | 84 % agreement (frames: 96 %); G points the right way (cos 0.92). Errors come from G, not from the motion. Without the I decay, I drifts into its clamp and agreement falls to 70 %. |
| [contrast-steps.md](../experiments/reports/contrast-steps.md) | Do separate ON/OFF steps make the I decay unnecessary? | They remove the drift of the mean (−73 → −0.5 steps), but I still spreads like a random walk without forgetting (agreement 0.53). The decay forgets accumulated per-event error. With decay: 0.836 → 0.839. |

## 3. End-to-end runs (C++)

| Check | How | Latest result |
|---|---|---|
| **Accuracy vs gyro**, events only (`--rotationFromTiming`) | `experiments/plot_r_vs_gyro.py output/<run>/R.txt`; live in the rotation plot | 60 s: cos median 0.992, mean 0.938; \|ω\| ratio 0.995 [0.86, 1.14]; per-axis corr 0.956 / 0.963 / 0.954 |
| Variants | same, with `--fuseR`, `--evidenceAccumulate`, `--onsetRule gap` | mean cos 0.941 / 0.918 / 0.933 |
| **Regression when the new mode is off** | compare R.txt with the run before the change | bit-identical |
| **Speed** | `Algorithm took … RTF` line, events loaded into memory first | timing mode RTF 0.26 (unpaced) |
| **Real-time display** | `--show --pace`, max lag line | 44 ms max lag with the rotation plot (27 ms without) |
| **Video** | `--writeFrames` + ffmpeg | `output/timing_video_plot/shapes_rotation_timing_plot.mp4` |

## Known gaps

- **No exact ground truth yet.** The gyro ignores translation (0.23 m/s median in this recording), which matters most at slow rotation. The simulator (M1) is needed for exact F and R, and for parameter sweeps.
- **The main loop in `main.cpp` has no unit test.** It is covered only by end-to-end runs (event order, IMU interleaving, display reads).
- **Known open issues seen in end-to-end runs:** |ω| ratio about 1.1 at slow and 0.93–0.95 at fast speeds; occasional spikes (mean cos below median).
- **G ↔ F in real units is untested**, because it does not exist yet (V has no time unit; see v-units.md).
- **The Python package `incomaps` has no tests yet**; the experiments are checked by their reports, not by assertions.
