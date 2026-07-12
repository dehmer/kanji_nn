Topic: Stroke clean-up metrics
Bot: ChatGPT
Date: 26 07 12

> Given this context, now the fun part: I want to clean-up/trim stroke vector data represented by above abstractions. The raw data is acquired through iPad/Apple pencil and Flutter Listener API (PointerEvent) which is know for micro-jitters and staircase patterns for slow movements. The main objective is to trim unwanted hook or event loops during the initial phase of pen touch down and the final phase just before life off. My idea currently is to calculate appropriate geometry/kinematic metrics (or features) to determine cut locations inside fixed (or dynamic) time windows. Are you up to lend me a hand with this?

Absolutely. This is the kind of problem where a handful of well-chosen geometric and kinematic features usually outperform trying to "smooth everything." I'd actually avoid thinking in terms of denoising first and instead think of it as **endpoint segmentation**: identify where the "real" stroke begins and ends.

From experience with stylus data, the artifacts tend to fall into a few categories:

* **Initial wobble**: tiny oscillations while the pen settles after contact.
* **Terminal hook**: a curved flick or loop just before lift-off.
* **Stationary jitter**: pen is essentially motionless but PointerEvents keep arriving.
* **Staircase motion**: quantization during very slow movement.
* **Pressure ramp**: pressure rises after contact and falls before lift-off.

Each leaves a different signature.

---

## I would compute a fairly rich feature set

I'd probably compute these for every sample.

### Kinematics

```text
dt
ds
speed = ds / dt

velocity_x
velocity_y

acceleration
jerk
```

where

```text
ds = ||p_i - p_{i-1}||
```

The absolute values aren't as interesting as the trends.

---

### Geometry

#### Tangent direction

```text
θ = atan2(dy, dx)
```

and then

```text
Δθ
```

using wrapped angles.

Large oscillations in Δθ are characteristic of endpoint jitter.

---

#### Curvature

Instead of finite differences of θ, I like using three-point curvature.

Given

```
A
B
C
```

compute the circumcircle radius.

```
κ = 1 / R
```

This is much less noisy.

---

#### Distance from origin/end

For trimming the beginning,

```
d_start(i)
```

For trimming the end,

```
d_end(i)
```

Sometimes the hook is literally revisiting previous positions.

---

#### Chord deviation

Within a local window

```
P_i ... P_j
```

measure

```
distance(point, line(P_i,P_j))
```

Hooks have much larger deviations than genuine straight starts.

---

### Pressure

Apple Pencil pressure is often surprisingly useful.

Compute

```
dp/dt
```

and maybe normalized pressure.

Many hooks occur while pressure is already dropping.

---

## Then compute window statistics

Rather than thresholding point-wise quantities, I'd compute statistics over a moving window.

For example

```
mean speed
median speed
std(speed)

mean curvature

direction variance

path length

displacement

path efficiency
```

---

### Path efficiency

One of my favorites.

```
efficiency =
    displacement /
    path_length
```

where

```
displacement =
||last-first||

path_length =
Σ ds
```

A straight stroke

```
≈ 1
```

A tiny loop

```
≈ 0
```

Hooks and endpoint scribbles have terrible efficiency.

---

## Endpoint detection

I would **not** try to identify the hook itself.

Instead identify

> "This is the first point that looks like intentional writing."

That's much easier.

Suppose you examine the first 80 ms.

For each candidate

```
i
```

compute the efficiency of

```
i → i+k
```

If

* speed becomes stable
* direction variance drops
* efficiency rises
* pressure stabilizes

then

```
cut = i
```

Do the same backwards for the end.

---

## I would use dynamic windows

Rather than

```
first 20 samples
```

I'd use

```
first 50–120 ms
```

Apple Pencil sampling isn't perfectly constant.

Time windows are more reproducible.

---

## I suspect one feature may dominate

I've done something similar with GPS traces.

The feature that ended up working astonishingly well was

```
path efficiency
```

Example

```
.
 \
  \
   \
```

Efficiency

```
0.99
```

versus

```
.
o
 \
```

or

```
.
↺
 \
```

Efficiency

```
0.2
```

because lots of distance is traveled with little net displacement.

Hooks are incredibly inefficient.

---

## Another feature I like

Instead of curvature alone:

```
signed turning angle
```

Accumulated over a window.

Imagine

```
→→→→
```

Sum

```
≈0°
```

Now

```
↺
```

Sum

```
250°
```

or

```
−180°
```

Tiny endpoint loops immediately stand out.

---

## Architecture

Given your `Stroke.clone()` API, I'd probably separate feature extraction from trimming:

```python
stroke = (
    stroke
    .clone(features=compute_velocity(stroke))
    .clone(features=compute_curvature(stroke))
    .clone(features=compute_direction(stroke))
    .clone(features=compute_efficiency(stroke))
    .clone(features=compute_turning_angle(stroke))
)

cut_start = detect_start(stroke)
cut_end   = detect_end(stroke)

trimmed = trim(stroke, cut_start, cut_end)
```

The nice thing is that the detector becomes completely independent of *how* the features are computed, and you can experiment by swapping in different detectors without touching the feature extraction.

---

If this were my project, I'd be tempted to go one step further and treat the start and end as small classification problems rather than thresholding individual features. For each point in the first (or last) ~100 ms, construct a feature vector—speed, efficiency, accumulated turning angle, local curvature, pressure trend, etc.—and assign a score for "this point belongs to the intentional stroke." The cut location is simply the first (or last) point where that score becomes convincingly high. It gives you a principled way to combine weak cues that, taken together, are often much more reliable than any single threshold.

> So far so good. I understand the notion of classification problem for specific regions, but for the time being I'd like to calculate all metrics for complete strokes to be able to use a multi-channel plot for visual inspection. As for the architecture, I'd rather use point-free style through function composition. I'd like to tackle each proposed metric one by one and we should keep in mind to align per-point metric results with initial raw data for possible 'hstacking' later on. Also metrics calculation should include intermediate results in feature dictionary for downstream usage where appropriate.

I like that direction. In fact, I'd postpone any trimming logic entirely until you have a good visualization pipeline. In signal processing, looking at the signals almost always suggests better heuristics than designing them a priori.

I also like the point-free/composition approach. Given your `Stroke` abstraction, I imagine something along these lines:

```python
pipeline = (
    velocity
    >> acceleration
    >> heading
    >> turning_angle
    >> curvature
    >> path_metrics
)
```

or simply

```python
stroke = (
    path_metrics(
        curvature(
            turning_angle(
                heading(
                    acceleration(
                        velocity(stroke)
                    )
                )
            )
        )
    )
)
```

where every stage is

```python
Stroke -> Stroke
```

and only appends features.

---

## I would establish a few conventions first

These will save a lot of headaches later.

### 1. Every per-point feature has `n_points` rows

This matches what `clone()` already enforces.

Scalar quantities become

```python
(n, 1)
```

not

```python
(n,)
```

For example

```python
speed.shape == (n, 1)
```

> I'm not too much concerned with (n, 1) or (n,) shape. np.column_stack() handles both cases gracefully as far as I know. Would you give me the code for velocity metric first?

I agree. `column_stack()` is perfectly happy with mixing `(n,)` and `(n, k)`, so there isn't much practical gain in forcing everything to `(n, 1)`. I mostly suggested it because some linear algebra code is happier that way. For feature storage and plotting, `(n,)` is completely fine.

I'd also propose a slight shift in terminology. Rather than a `velocity()` metric, I'd call the first stage `kinematics()`, because velocity computation naturally yields several fundamental quantities that later stages will reuse.

```python
def kinematics(stroke: Stroke) -> Stroke:
    t = stroke.t
    xy = stroke.xy

    # First-order differences, aligned with the current sample.
    dt = np.diff(t, prepend=np.nan)
    dxy = np.diff(xy, axis=0, prepend=np.full((1, 2), np.nan))

    dx = dxy[:, 0]
    dy = dxy[:, 1]

    ds = np.hypot(dx, dy)

    with np.errstate(divide='ignore', invalid='ignore'):
        velocity = dxy / dt[:, None]
        speed = ds / dt

    return stroke.clone(features={
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "ds": ds,
        "velocity": velocity,
        "speed": speed,
    })
```

A few remarks:

* `np.diff(..., prepend=np.nan)` keeps everything aligned with the original samples, which I think is the cleanest convention.
* `np.errstate()` suppresses warnings for the first sample and for any duplicate timestamps. The resulting `NaN`/`inf` values are informative and can be handled downstream if necessary.
* `velocity` is `(n, 2)`, while the scalar features are `(n,)`.

One thing I'd actually like to discuss before we proceed further is **the alignment convention**. There are three reasonable choices:

1. **Current-point alignment** (used above):

   ```
   v[i] = (P[i] - P[i-1]) / dt
   ```

   `v[0] = NaN`.

2. **Forward alignment**:

   ```
   v[i] = (P[i+1] - P[i]) / dt
   ```

   `v[-1] = NaN`.

3. **Centered differences**:

   ```
   v[i] = (P[i+1] - P[i-1]) / (t[i+1] - t[i-1])
   ```

   Endpoints become `NaN`.

For visualization and endpoint trimming, I'm actually leaning toward **centered differences**. They produce noticeably cleaner velocity, heading, and curvature estimates without any explicit smoothing, and the two undefined endpoint values aren't an issue since those are exactly the regions where derivatives are least reliable anyway.

Since many subsequent metrics (heading, turning angle, curvature) depend on this choice, it's worth settling on a convention now rather than revisiting it later.

> Choosing an appropriate alignment is a very good point. I was not consciously aware of its implications. We should go with centered differences then. Which won't affect velocity and speed, right?

Exactly—and I'm glad you questioned it, because there are subtle trade-offs.

Strictly speaking, centered differences **do** affect the numerical values of velocity and speed. They are different estimators:

Forward difference:

[
v_i = \frac{p_i - p_{i-1}}{t_i - t_{i-1}}
]

Centered difference:

[
v_i = \frac{p_{i+1} - p_{i-1}}{t_{i+1} - t_{i-1}}
]

The latter is a second-order approximation and is generally less sensitive to measurement noise.

### However...

For *your* application, I would actually **not** use centered differences for the first derivative.

Here's why.

Suppose you have a genuine endpoint artifact:

```
P0  P1  P2  P3  P4
o---o~~~o========>
```

where `P0..P2` are the initial wobble.

A centered estimate at `P2` uses **P1 and P3**.

That means it already "looks into" the genuine stroke. In other words, the derivative around the transition is contaminated by information from both sides of the cut.

Forward/backward differences are more local:

```
v2 = P2 - P1
```

Only the artifact contributes.

This is actually desirable if your eventual goal is to detect **where** the artifact ends.

---

## My current thinking

I'd separate two notions:

### Local features

Used for endpoint detection.

* `dx`
* `dy`
* `ds`
* `speed`
* `heading`

computed with one-sided differences.

These describe what happened **between adjacent events**.

---

### Differential features

Used for geometry.

* acceleration
* turning angle
* curvature

computed with centered differences.

These benefit enormously from the increased accuracy.

---

Notice that curvature is really a derivative **of a derivative**, so that's where centered stencils shine.

---

## Another observation

Apple Pencil samples at roughly 120–240 Hz depending on hardware.

At those rates,

```
Pi-1
Pi
Pi+1
```

are separated by only a few milliseconds.

Your hooks and loops, on the other hand, last tens of milliseconds.

So using one-sided differences for speed isn't particularly noisy in practice, while it preserves causality near the endpoints.

---

I think we should also distinguish between **segment quantities** and **point quantities**.

For example:

```
P0 ---- P1 ---- P2 ---- P3
   S0      S1      S2
```

* `dx`, `dy`, `ds`, and even instantaneous speed are fundamentally properties of **segments**.
* Heading, turning angle, and curvature are fundamentally properties of **points** (or vertices).

Trying to force everything into the same numerical differentiation scheme can blur that distinction.

So if I were designing this library from scratch today, I'd likely compute:

* segment features (`dx`, `dy`, `dt`, `ds`, `speed`) with adjacent samples,
* point features (`heading`, `turning_angle`, `curvature`) with centered stencils.

That gives each quantity the interpretation that's most natural mathematically and, I suspect, most useful for the endpoint classification problem you're ultimately trying to solve.

> Flutter Pointer events are sample with 8ms to 14ms on average. kinematics() function already has one-sided alignment (prepend np.nan), so this won't change, correct?

Correct. Given what you've told me about the sampling rate and your end goal, I would leave `kinematics()` exactly as it is.

In fact, after thinking about it more, I think we should adopt a simple rule:

> **Features that describe segments use one-sided differences. Features that describe points use centered differences.**

That gives every feature a clear geometric interpretation.

| Feature         | Meaning                       | Alignment |
| --------------- | ----------------------------- | --------- |
| `dt`            | elapsed time                  | segment   |
| `dx`, `dy`      | displacement                  | segment   |
| `ds`            | arc length increment          | segment   |
| `velocity`      | average velocity over segment | segment   |
| `speed`         | average speed over segment    | segment   |
| `heading`       | tangent direction             | point     |
| `turning_angle` | change of tangent             | point     |
| `curvature`     | local geometry                | point     |

This also fits your storage convention nicely:

```text
P0 ---- P1 ---- P2 ---- P3
   dx1    dx2    dx3
```

where `dx[i]` means "the segment ending at sample `i`".

Then later

```text
P0 ---- P1 ---- P2 ---- P3
        θ1
```

where `heading[1]` is computed from the neighborhood around `P1`.

That distinction becomes surprisingly useful when interpreting plots.

---

One more suggestion that I think will pay off later: since you're already storing intermediate quantities, I'd have `kinematics()` produce **only primitive quantities**:

```python
dt
dx
dy
ds
velocity   # (n, 2)
speed
```

and **not** heading.

Then `heading()` becomes the first function to intentionally use a centered stencil, taking `velocity` (or directly `x`, `y`) as input. From that point on, all geometric descriptors—turning angle, curvature, even osculating radius—can consistently build on the point-based convention.

I think that's a very clean separation between **motion along the path** (segment-based) and **shape of the path** (point-based), and it should make the multi-channel plots much easier to interpret.

> Can you interpret plot images? If so I can upload the first multi-channel plot.

This is already a very informative visualization. Even with just pressure, velocity, and speed, I can see a few things that support your original hypothesis.

## 1. The initial touch-down region

The red vertical line appears to mark your intended cut location.

Before it:

* pressure is low and ramping up
* speed is almost zero
* velocity components oscillate around zero

That looks much more like **contact stabilization** than intentional writing.

Interestingly, the spatial plot also agrees: there is almost no visible movement until the stroke starts moving rightwards.

So far, speed + pressure alone are already separating "pen settling" from "writing."

---

## 2. The middle of the stroke

The speed profile is pleasantly smooth.

The large spike around 220 ms immediately caught my eye.

Because neither pressure nor the trajectory shows anything dramatic there, my first suspicion is that this is a sampling artifact:

* one unusually large Δx,
* or one unusually small Δt.

That makes me want to see **dt** on the plot.

If dt suddenly drops from 10 ms to 5 ms, speed doubles without any geometric reason.

---

## 3. The end of the stroke

This is the interesting part.

Near 520–600 ms:

* speed rises again
* velocity changes direction
* pressure is still almost constant

Looking at the trajectory, that corresponds exactly to the little hook.

That tells me something important:

> The hook is **not** characterized by low speed.

It is characterized by **changing direction**.

That reinforces my earlier thought that turning angle / heading will probably be much more discriminative than speed.

---

## 4. Pressure

Pressure is cleaner than I expected.

It has three obvious phases:

```
low
 ↑
stable plateau
 ↓
lift-off
```

The pressure derivative might become surprisingly useful later.

Not because it identifies hooks, but because it defines a confidence interval:

> "I'm somewhere inside the writing."

---

# What I'd add next

Not curvature.

Not acceleration.

I'd add **heading**.

Specifically

```python
heading = atan2(vy, vx)
```

and **unwrap** it.

If you don't unwrap, you'll get jumps like

```
179°
-179°
```

instead of

```
179°
181°
```

NumPy already provides

```python
heading = np.unwrap(np.arctan2(vy, vx))
```

I have a strong suspicion the hook will become immediately obvious.

---

# Then

From heading,

```
turning_angle = d(heading)
```

computed with centered differences.

I wouldn't even bother with curvature until we've seen heading.

---

# One thing I noticed

Your x-axis says

```
Timestamp
```

but it appears to be sample index.

The tooltip shows

```
Time: 129.000 s
```

yet the trace ends around 600.

I'm guessing those are milliseconds from stroke start?

If so, I'd label the axis

```
Time (ms)
```

or

```
Elapsed time (ms)
```

Otherwise someone reading the plot later will assume seconds.

---

## One final thought

I think your instinct to compute *all* features for the whole stroke before designing the trimming algorithm is exactly right. Endpoint artifacts often reveal themselves through combinations of signals, and you can't know which combinations matter until you've looked at a variety of strokes.

At this point, though, I'd slightly revise my earlier priority list based on this plot:

1. **Heading** (most promising)
2. **Turning angle**
3. **dt** (to identify sampling irregularities)
4. **Curvature**
5. **Path efficiency** (computed over a sliding window)

The nice thing is that each one builds naturally on the previous, and I wouldn't be surprised if by the time we have heading and turning angle, the terminal hook almost "lights up" in the plots.

> As for the spike at 220ms, Although input is smoothened with gaussian 1D (sigma = 1.0) I guess the data might still contain volatile segments. I updated 'Timestamp' to 'Elapsed time (ms)'. If I remember correctly, you proposed not to smooth at all and process raw data. Would you like to see the same plot without smoothing?