Topic: Stroke Segmentation
Bot: ChatGPT
Date: 26 07 09

> Do you know of any algorithmic approach to classify and segmentize individual kanji strokes (in vector form.) Of special interest would be to detect unwanted noise at stroke starts and endings stemming from digital writing on tablets, e.g. iPad/Apple Pencil.

Yes. This is actually a fairly active topic in online handwriting recognition and digital ink processing. The good news is that because you're working with **vector strokes** (sampled pen trajectories) rather than raster images, you have much richer information available: time, velocity, pressure, tilt, and stroke ordering. That makes endpoint cleanup much more tractable.

The problem can be viewed as two related tasks:

1. **Segmenting the trajectory into intentional stroke primitives**
2. **Classifying endpoint artifacts vs. intended brush motion**

For kanji, the second is especially interesting because many legitimate strokes intentionally have hooks, flicks (*hane*), stops (*tome*), or sweeps (*harai*), which can superficially resemble tablet jitter.

## 1. Model the stroke as a signal

Suppose a stroke is

[
p_i=(x_i,y_i,t_i,p_i,\theta_i,\phi_i)
]

where

* position
* timestamp
* pressure
* tilt
* azimuth

From this derive

* velocity
* acceleration
* curvature
* angular velocity
* pressure derivative

These signals are often more informative than geometry alone.

---

## 2. Endpoint detection via velocity profiles

A surprisingly effective observation:

Digital pen noise usually occurs when

* velocity approaches zero
* pressure changes rapidly
* tiny oscillations appear
* curvature spikes

A typical endpoint looks like

```
------real stroke------
                    \
                     \__
                      /\_
                     /  \
```

The little oscillations have

* very low speed
* alternating curvature
* very short arc length

whereas a true *hane* has

* smooth acceleration
* increasing speed
* continuous curvature.

---

## 3. Fit-and-trim algorithms

One robust approach is iterative fitting.

### Step A

Fit the stroke with

* cubic B-splines
* clothoids
* Bézier chains

using robust regression.

### Step B

Measure residuals.

Noise tends to produce

* tiny loops
* high-frequency oscillation

that contribute almost nothing to the overall fit.

Then trim endpoints until

* RMS error stops improving.

This is similar to robust polyline simplification.

---

## 4. Curvature-scale space

Another elegant method is curvature scale-space analysis.

Compute curvature

[
\kappa(s)
]

along arc length.

Then smooth with increasing Gaussian scales.

True structural features persist across scales.

Tiny endpoint wiggles disappear almost immediately.

So endpoint noise corresponds to extrema that vanish under slight smoothing.

---

## 5. Hidden Markov Model

Many online handwriting systems model a stroke as

```
START
 ↓
Acceleration
 ↓
Steady motion
 ↓
Deceleration
 ↓
END
```

Endpoint artifacts become additional states

```
START
 ↓
Noise
 ↓
Stroke
 ↓
Noise
 ↓
END
```

The Viterbi algorithm can infer the most likely segmentation.

This was common in older handwriting recognizers.

---

## 6. Bayesian endpoint classifier

If you have examples, construct features like

* endpoint length
* average speed
* pressure drop
* curvature variance
* turning angle
* local PCA eigenvalues
* number of inflection points

Then classify

```
intentional ending

vs

tablet artifact
```

using

* Random Forest
* XGBoost
* SVM

This often outperforms heuristic rules.

---

## 7. Douglas–Peucker isn't enough

People often try

* Douglas–Peucker
* Visvalingam simplification

These reduce vertices but don't distinguish

```
hook

vs

endpoint wobble
```

because both may have similar geometric deviation.

You need temporal information.

---

## 8. Elastic energy

Treat the stroke as a thin elastic rod.

Noise contributes enormous bending energy over tiny arc length.

Compute

[
E=\int \kappa^2 ds
]

Near endpoints,

trim until

* total elastic energy decreases substantially
* endpoint position changes little.

This works remarkably well.

---

## 9. Learning from kanji stroke grammar

This is probably the most powerful approach if you're specifically targeting kanji.

Kanji strokes belong to a relatively small set of terminal types:

* stop (*tome*)
* hook (*hane*)
* sweep (*harai*)
* dot (*ten*)
* horizontal
* vertical
* left-falling
* right-falling

A classifier trained on these endpoint types can distinguish

```
true hook

from

pen wobble
```

because

* hook direction
* curvature evolution
* velocity profile
* pressure evolution

are all highly stereotyped.

This is much easier than general handwriting recognition.

---

## 10. Stroke grammar / graph optimization

An advanced formulation is:

Each stroke is a graph.

Generate multiple candidate trims near both endpoints.

For each candidate compute

```
shape likelihood
+
endpoint likelihood
+
kanji grammar prior
+
smoothness
```

Then optimize globally (dynamic programming or graph search).

This resembles approaches used in digital calligraphy software.

---

# What I'd recommend for an Apple Pencil dataset

Given Apple Pencil data (120–240 Hz, position, pressure, tilt), I'd use a hybrid pipeline:

1. Resample uniformly by arc length (e.g. every 0.2–0.5 mm).
2. Apply light smoothing (e.g. Savitzky–Golay or cubic spline) to preserve corners while reducing sensor noise.
3. Compute local features:

   * speed
   * acceleration
   * curvature
   * pressure
   * pressure derivative
   * turning angle
   * arc length from each endpoint
4. Generate candidate trims by progressively removing the first/last (k) points (or a small endpoint arc-length window).
5. Score each candidate using a weighted objective that rewards:

   * low curvature variance near the endpoint,
   * monotonic pressure decrease for stroke endings,
   * smooth velocity profile,
   * preservation of the interior stroke geometry.
6. If labeled data is available, replace the hand-tuned score with a classifier (e.g. gradient-boosted trees) trained to predict the optimal trim.

This approach is computationally lightweight, explainable, and generally robust to the small overshoots and hesitations that occur when lifting an Apple Pencil.

If your goal is **high-quality digital calligraphy** rather than handwriting recognition, I'd also consider explicitly modeling the canonical endpoint types (*tome*, *hane*, *harai*) and treating any endpoint behavior that doesn't fit one of those models as a likely artifact. That leverages the constrained structure of kanji and typically performs better than generic trajectory denoising.
