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
| R → F | `update_FR` | all pixels, `updateIterationsFR` times per `timeStep` |
| F → R | `update_RF` | incremental least squares (running sums `A`, `B`) |
| gyro → R | `update_RIMU` | per IMU sample |
| frame → I | `update_Ifusion` | per image |

Per event, `event_step` applies V → I, then I → G around the pixel, then G → I, then F/G/R updates in a shuffled order.

## Deliberate deviations from the paper

Kept as defaults; each can be switched off (field in `Parameters`, command-line option in brackets):

| Deviation | Switch | Off means |
| --- | --- | --- |
| **V → I injection.** Each event adds `weight_IV · V` to `MI`, clipped to [`minPotential`, `maxPotential`]. The paper never observes I directly. | `weight_IV` | `weight_IV = 0` |
| **Decay.** `MI` decays exponentially towards `neutralPotential` with time constant `decayParam`. | `useDecay` (`--decay`) | no decay |
| **Clamping.** F and G are clipped to ±`gamma` and entries below `eps` are zeroed after each update. | `clampMaps` (`--clamp`) | no clipping, no zeroing |
| **Per-frame G reset** when built with `IMAGES`: G is zeroed after each output frame. | `resetGEachFrame` (`--resetG`) | G keeps its state |
| **Central differences** for ∇I on an H × W grid instead of forward differences on (H+1) × (W+1). | none | — |

## Known issues

See M0 in `TASKS.md` for the full list and status. Open design question: R has no physical time unit. V is a per-event contribution, not a temporal derivative per bin, so F (and R) are "pixels per event contribution". Fusing a gyro (rad/s) into R therefore needs an explicit scale that the current model does not define.
