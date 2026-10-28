# Roadmap and principles

## Roadmap

```
WP1  Virtual DVS128 → NumPy reference (+gyro config) → generic engine → optimised sync (C++)
       → [gate: matches reference, real time] → asynchronous → [gate: same accuracy, lower latency]
       → gyro experiments
WP2  Sensor setup (can start in parallel)
       → frame observation + confidence maps → gated relations + full IMU
       → depth, translation, ownership → temporal prediction, shading split
     Each stage: ablation, observability, failure map
```

## Expanding the network

Each new quantity becomes a map tied to existing maps by the relation that defines it. Candidates, roughly by payoff:

- **Depth D and translation T** by splitting flow into rotation part S and parallax part P. Required once the platform translates.
- **IMU maps:** gyro nudges R, accelerometer nudges T. Both weak inputs, like V.
- **Ownership map:** assigns each pixel to one of several rigid motions (several R/T pairs).
- **Confidence maps,** one per state map; they drive reliability weighting.
- **Shading vs. reflectance split** of I.
- **Temporal prediction:** previous state propagated by F, as a relation (turns warm-starting into part of the network).

## Principles from human multisensory perception

1. **Reliability weighting.** Humans combine cues close to statistically optimal, weighting each by its precision (Ernst & Banks 2002). Here: δ scales with the confidence of the source map; strong where |G| is large and V clean, near zero in flat regions.
2. **Causal inference.** The brain first decides whether two signals share a cause and fuses them only if so (Körding et al. 2007). Here: relations switch themselves off when their residual stays large (robust losses, ownership map).
3. **Priors.** Many motion illusions follow from a "slow and smooth" prior plus the aperture problem (Weiss, Simoncelli & Adelson 2002). Here: smoothness and rigid motion become explicit, weighted maps.
4. **Efference copy.** Self-motion is predicted from the motor command before the senses report it. Here: available motor commands enter as a weak input to R and T.

## Finding the limits

- **Observability analysis.** Linearise the total residual around a solution; its null space shows what can never be determined (β scale of I; common scale of depth and translation with one camera). Cramér–Rao bounds give the best achievable accuracy per pixel.
- **Sweeps in simulation.** Vary texture density, contrast, speed, noise, motion type → failure map of operating conditions.
- **Ablations.** Remove one sensor or relation at a time.

## What works without object recognition

Insect-level vision, provided a sensor supplies metric scale (IMU accelerometer, known camera height, stereo, defocus):

- egomotion (rotation, heading from the focus of expansion, metric translation with an IMU)
- time-to-contact (image size ÷ expansion rate, Lee 1976), obstacle detection, collision avoidance
- relative depth and depth order; ground plane and surface orientation
- independent-motion detection, figure–ground segmentation by common motion
- tracking, stabilisation, vibration analysis
- HDR and deblurred images from fused events and frames

Out of reach: object identity, absolute size from one camera without a metric sensor, anything depending on what an object is or is for. There the network is a fast front end whose maps feed a recognition model.
