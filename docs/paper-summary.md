# Interacting Maps – condensed paper summary

**Source:** M. Cook, L. Gugelmann, F. Jug, C. Krautz, A. Steger, "Interacting Maps for Fast Visual Interpretation", Proc. IJCNN 2011, San Jose, pp. 770–776 (INI and ETH Zurich).

A small network of recurrently coupled, retinotopic maps relaxes from raw DVS events to a mutually consistent scene interpretation (intensity, gradient, optical flow, camera rotation), with no feed-forward pipeline and no designated output.

- **Input is weak by design.** Binned DVS events only nudge the temporal-derivative map V. The intensity image I is never observed; it is inferred together with everything else.
- **Every constraint is weak alone.** Per pixel, the flow constraint removes only 1 of 4 degrees of freedom in F and G. A solution emerges only through recurrence across all maps.
- **Main claim:** a single binned event frame is enough to recover both a grayscale image and the optical flow, which a single conventional frame cannot provide.
- **Motivation:** mimic cortical architecture (strong recurrence between areas, weak but decisive feed-forward input) rather than a filter pipeline.

## Maps and relations

Six maps are tied by three per-pixel relations. Each relation follows from the definitions of the maps it connects.

| Map | Meaning | Type | Size (paper) | Role |
| --- | --- | --- | --- | --- |
| V | temporal intensity derivative | scalar | 128 × 128 | sole input (binned events) |
| I | light intensity | scalar | 129 × 129 | inferred |
| G | spatial intensity gradient | 2D vector | 128 × 128 | inferred |
| F | optical flow | 2D vector | 128 × 128 | inferred |
| C | pixel viewing direction | 3D vector | 128 × 128 | constant, from calibration |
| R | camera rotation | one 3D vector | 1 | inferred |

Optical flow constraint (links V, F, G):

```latex
-V_{x,y} = \vec{F}_{x,y} \cdot \vec{G}_{x,y}
```

Gradient definition (forward differences, links I, G):

```latex
\vec{G}_{x,y} = \nabla I_{x,y} = \begin{pmatrix} I_{x+1,y} - I_{x,y} \\ I_{x,y+1} - I_{x,y} \end{pmatrix}
```

Rotational flow (links F, R, C; m32 projects world space to image space):

```latex
\vec{F}_{x,y} = m_{32}(\vec{R} \times \vec{C}_{x,y})
```

**Only two relations couple pixels.** G = ∇I couples neighbours; F = R × C couples all pixels through the single R. The flow constraint is purely local.

**Scale ambiguity β.** Scaling G and I by β and F and R by 1/β leaves V unchanged, for any non-zero β including negative ones. A slow, high-contrast scene and a fast, low-contrast scene look identical; so does inverted contrast moving the opposite way. The paper resolves β arbitrarily in the first frame and keeps it stable by warm-starting each frame from the previous converged state.

**Factor-graph view.** The network is a simplified loopy factor graph: maps are variable sets, relations are factors (e.g. I and F are conditionally independent given G). At 128 × 128 that is over 60,000 variables, so the authors keep point estimates rather than running belief propagation over distributions.

## Update rules

Every map is updated by the same pattern: assume the other maps in a relation are correct, compute the nearest **candidate** map that satisfies it, then take a small step towards that candidate (relaxation on local constraints).

```latex
X \leftarrow (1-\delta_{AB})\,X + \delta_{AB}\,X_{\text{candidate}}
```

Each direction of each relation has its own rate δ (e.g. δ_IG for I → G, δ_GI for G → I). The paper does not report the δ values used.

**V, F, G.** The constraint cannot be inverted per pixel (aperture problem): given V and G, valid F lie on a line perpendicular to G. Use the quadratic error and its gradient, or project F directly onto that line; both give the same direction. G → F and F → G are symmetric.

```latex
Q_{V,G}(\vec{F}_{x,y}) = (V_{x,y} + \vec{F}_{x,y} \cdot \vec{G}_{x,y})^2, \qquad \frac{dQ}{d\vec{F}_{x,y}} = 2\,\vec{G}_{x,y}\,(V_{x,y} + \vec{F}_{x,y} \cdot \vec{G}_{x,y})
```

**I → G.** Step G towards the forward-difference gradient of I.

```latex
\vec{G}_{x,y} \leftarrow (1-\delta_{IG})\,\vec{G}_{x,y} + \delta_{IG}\,\nabla I_{x,y}
```

**G → I.** Form the residual Ψ = G − ∇I. Because I at (x, y) also enters the gradients of (x−1, y) and (x, y−1), take backward differences of Ψ (out-of-bounds entries set to 0), then step I. This equals gradient descent on the squared residual.

```latex
\hat{\Psi}^{(x)}_{x,y} = \Psi^{(x)}_{x,y} - \Psi^{(x)}_{x-1,y}, \qquad I_{x,y} \leftarrow (1-\delta_{GI})\,I_{x,y} + \delta_{GI}\,(I_{x,y} - \hat{\Psi}^{(x)}_{x,y} - \hat{\Psi}^{(y)}_{x,y})
```

(With the forward-difference convention above, ∂(½‖G − ∇I‖²)/∂I = Ψ̂^(x) + Ψ̂^(y), so the candidate I − Ψ̂^(x) − Ψ̂^(y) is one unit gradient-descent step. Verify against a finite-difference gradient in a unit test.)

**R → F.** Step each flow vector towards the flow the global rotation predicts.

```latex
\vec{F}_{x,y} \leftarrow (1-\delta_{RF})\,\vec{F}_{x,y} + \delta_{RF}\,m_{32}(\vec{R} \times \vec{C}_{x,y})
```

**F → R.** Each flow vector restricts R to a line through C × m23(F), parallel to C (m23 lifts image vectors to world space). Fit R to all lines by linear least squares, then step.

```latex
d^2_{\vec{R},x,y} = |\vec{R} - \vec{C}_{x,y} \times m_{23}(\vec{F}_{x,y})|^2 - (\vec{R} \cdot \vec{C}_{x,y})^2, \qquad \vec{R} \leftarrow (1-\delta_{FR})\,\vec{R} + \delta_{FR}\,\arg\min_{\vec{R}} \sum_{x,y} d^2_{\vec{R},x,y}
```

(The distance formula assumes unit-length C.)

## Results and limitations

On live DVS data the network converges to correct rotation and translation estimates at about 10 frames per second, but the evaluation is qualitative only.

**Setup.** 128 × 128 DVS (Lichtsteiner, Posch and Delbruck 2006). Events binned over a short window (length not reported) into V frames. Each frame relaxed to a stable state, all maps warm-started from the previous frame (random at the start). C++ on a 2.66 GHz mobile CPU, about 10 frames per second.

**Results (camera over a keyboard).** A rotation centred in the upper-left quadrant and a left-to-right translation were identified correctly. I reproduced the keyboard's shading from sparse events, including regions with almost no events (information propagated from neighbours). No explicit smoothing or outlier clipping was needed.

**Limitations to address when extending.**

- **Local optima from random initialisation.** A rotation about the image centre can converge to uniform leftward flow (β varying with vertical position). Warm starting avoids this between frames, but not at start-up.
- **Restricted world model.** Static scene, rotation-only camera, fixed calibration C, a single rigid motion.
- **No quantitative evaluation.** No error metrics, ground truth or baselines.
- **Frame-based, not spiking.** Events are binned into frames; an asynchronous spiking version is listed as future work.
- **Unreported parameters.** Bin length, step sizes δ and iterations to convergence are not given.

## Extension points named by the authors

- **Structure from motion:** split flow into F = S + P. S is the image shift from rotation R; P is the parallax from camera translation, tied to a depth map D and a global translation vector T.
- **Multiple moving objects:** several rotation vectors instead of one R.
- **Self-calibration:** let C be updated by the network instead of held constant.
- **Spiking implementation:** asynchronous, event-driven updates on neuromorphic hardware (SpiNNaker, BrainScaleS and similar are cited).
