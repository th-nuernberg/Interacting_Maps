# Event-based C++ implementation

The C++ code in `src/` and `include/` predates the specs in this folder. It runs the paper's update rules **per event** instead of per binned frame. This file maps it onto `paper-summary.md`, lists where it deviates on purpose, and records known issues. Fixes are tracked under M0 in `TASKS.md`.

## Map layout

| Paper | C++ (`OpticalFlowState`) | Shape | Notes |
| --- | --- | --- | --- |
| V | `V` (scalar per event), `V_Vis` (display only) | H × W | `V = polarity · eventContribution`, polarity ∈ {−1, +1} |
| I | `MI` | H × W | Not (H+1) × (W+1); see deviations |
| G | `G` | H × W × 2 | channel 0 = y, channel 1 = x |
| F | `F` | H × W × 2 | channel 0 = y, channel 1 = x |
| C | `CCM`, with tangents `dCdx`, `dCdy` | H × W × 3 | 3-vectors stored as (y, x, z); pinhole from fx, fy, cx, cy, no distortion |
| R | `R` | 3 | stored as (y, x, z) |

Camera frame: x right, y down, z forward (right-handed), as in `CLAUDE.md`. 3-vectors are stored in the order (y, x, z); `crossProduct3x3` and `crossProduct1x3` permute indices so that they compute the right-handed cross product of the underlying (x, y, z) vectors. F and G channel 0 point down (increasing row). R is the paper's R, i.e. minus the camera rotation; the test "F = m32(R x C) with R = -(camera rotation) reproduces projected pixel motion" pins this down against explicit pinhole projection.

## Update functions

Names read "target from source": `update_FG` updates F from G. This is the reverse of the `delta_AB` convention in `CLAUDE.md` (A → B); the C++ names are kept to avoid churn.

| Paper direction | Function | Scope |
| --- | --- | --- |
| G, V → F | `update_FG` | event pixel |
| F, V → G | `update_GF` | event pixel |
| I → G | `update_GI` | event pixel and its 4 neighbours |
| G → I | `updateGIDiffGradient` + `update_IG` | event pixel |
| R → F | `relaxB` + `relaxFR` (default, `--asyncFR`) or `update_FR` | event-driven closed form (see below); synchronous: all pixels, `updateIterationsFR` times per `timeStep` |
| F → R | `update_RF` | incremental least squares (running sums `A`, `B`) |
| gyro → R | `update_RIMU` | per IMU sample |
| frame → I | `update_Ifusion` | per image |

Per event, `decayNeighbourhood` first decays every pixel within Manhattan distance 2 of the event to the event time (decay is lazy; the gradient stencils of `event_step` read all of them). `event_step` then applies V → I, then I → G around the pixel, then G → I, then F/G/R updates in a shuffled order.

## Deliberate deviations from the paper

Kept as defaults; each can be switched off (field in `Parameters`, command-line option in brackets):

| Deviation | Switch | Off means |
| --- | --- | --- |
| **V → I injection.** Each event adds `weight_IV · V` to `MI`, clipped to [`minPotential`, `maxPotential`]. The paper never observes I directly. | `weight_IV` | `weight_IV = 0` |
| **Decay.** `MI` decays exponentially towards `neutralPotential` with time constant `decayParam`. | `useDecay` (`--decay`) | no decay |
| **Clamping.** F and G are clipped to ±`gamma` and entries below `eps` are zeroed after each update. | `clampMaps` (`--clamp`) | no clipping, no zeroing |
| **Per-frame G reset** when built with `IMAGES`: G is zeroed after each output frame. Old workaround for stale G, superseded by silent-time relaxation below; **off by default**. | `resetGEachFrame` (`--resetG`) | G keeps its state |
| **Central differences** for ∇I on an H × W grid instead of forward differences on (H+1) × (W+1). | none | — |

## Pixels without events (asynchronous V = 0)

In the paper every pixel is updated every iteration, and a pixel without events has V = 0: an observation ("brightness did not change"), not missing data. Event-driven updates alone never touch such pixels, so G stayed wherever the last edge left it.

Instead of a per-bin sweep, each pixel stores `GTimeSurface`, the time up to which its G is current. Before `event_step` reads G at the event pixel and its 4 neighbours, `relaxSilentNeighbourhood` integrates the two relations the paper applies at a silent pixel over the elapsed time, in closed form:

- G ← I: G relaxes towards g = ∇I (from the freshly decayed stencil) at rate `rate_GI_silent`.
- G ← F with V = 0: G's component along F decays at rate `rate_GF_silent`.

Split along F, the perpendicular part goes to g⊥ with rate `rate_GI_silent`, and the parallel part goes to `rate_GI / (rate_GI + rate_GF) · g∥` with rate `rate_GI + rate_GF`. The solution is exact for g and F constant over the silent time; both are approximations (an event two pixels away can change ∇I, and global F ← R updates change F). Output frames show `relaxedSilentCopy(G)`, every pixel relaxed to the frame time, without changing the state.

The rates default to `-ln(1 - weight) / time_step`, which reproduces one paper update per time step (tested). They are open parameters, to be determined by the M2 sweeps. Switch: `silentRelaxation` (`--silentG`). F is not relaxed over silent time; it is regularised globally by F ← R.

## Event-driven F ← R

The synchronous version applies `update_FR` `updateIterationsFR` times to every pixel each `timeStep` and then rebuilds the F → R sums, O(pixels) at 100 Hz, which dominated the run time. In the event-driven version (`asyncFR`, default):

- F at a pixel relaxes towards m32(R × C) at rate `rate_FR = -updateIterationsFR · ln(1 - weight_FR) / timeStep` (one time step reproduces the synchronous passes; tested). `relaxFR` applies this lazily, with the current R, where an event reads F (event pixel and its 4 neighbours) and for display (`relaxedFRCopy`).
- B = Σ (I − CCᵀ)(C × m23(F)) is linear in F, and a relaxed pixel contributes (I − CCᵀ)R. So B relaxes in closed form, B ← A R + (B − A R) e^(−rate_FR dt), in O(1) per event (`relaxB`). This is exact because R only changes at events. `relaxFR` refreshes `old_points` without touching B.
- After each event step, `register_R_point` puts the final F of the event pixel into B (F ← G may run after F → R within the step).

Approximation: `relaxFR` uses the current R for a pixel's whole silent time, while B integrated the actual R history. The two differ only through R changes within the last few 1/rate_FR (≈ 1.6 ms).

Output frames, the live window (`--show`) and `R.txt` samples run at `--fps` (default 60), independent of `timeStep`.

## Known issues

See M0 in `TASKS.md` for the full list and status. R does not yet track the camera rotation: on shapes_rotation, the correlation of R with the gyro is below 0.25 in both the synchronous and the event-driven version (2026-10-30). Open design question: R has no physical time unit. V is a per-event contribution, not a temporal derivative per bin, so F (and R) are "pixels per event contribution". Fusing a gyro (rad/s) into R therefore needs an explicit scale that the current model does not define. `update_RIMU` takes it as `Parameters::imuScale` (default 1, i.e. unscaled) together with the camera–IMU rotation `R_cam_imu` (default identity); both must be set before `weight_RIMU > 0` gives meaningful results.
