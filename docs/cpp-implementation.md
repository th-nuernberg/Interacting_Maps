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

3-vectors are stored in the order (y, x, z). `crossProduct3x3` and `crossProduct1x3` permute indices so that they compute the right-handed cross product of the underlying (x, y, z) vectors.

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

These are kept as defaults and are to be made switchable (M0):

- **V → I injection with decay.** Each event adds `weight_IV · V` to `MI`, which decays exponentially towards `neutralPotential` with time constant `decayParam`. The paper never observes I directly.
- **Clamping.** F and G are clipped to ±`gamma` and values below `eps` are zeroed after each update.
- **Central differences** for ∇I on an H × W grid instead of forward differences on (H+1) × (W+1).
- **Per-frame G reset** when built with `IMAGES` (visualisation build).

## Known issues

See M0 in `TASKS.md` for the full list and status. Open design question: R has no physical time unit. V is a per-event contribution, not a temporal derivative per bin, so F (and R) are "pixels per event contribution". Fusing a gyro (rad/s) into R therefore needs an explicit scale that the current model does not define.
