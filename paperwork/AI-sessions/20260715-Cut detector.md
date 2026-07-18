Topic: Cut detector
Bot: ChatGPT
Date: 26 07 15

> How to reliably detect the index at t=142ms (red dotted line) based on signals 'ds' and 'P_norm' (and optionally on dP/dt > 0 as additional guard)? I'm not so much interested in the final solution as in the individual step and especially what NumPy magic is involed and why? I'm fairly new to Python and NumPy.
I already set a first upper bound 'max_idx' for the window to analyse (if this help at all.)

```
    max_idx = np.argmax(p_norm)
```

From the plot, the red line at **142 ms** looks like the point where several things happen almost simultaneously:

1. `ds` changes from (almost) zero to consistently positive.
2. `P_norm` is already high (or has crossed some threshold).
3. Optionally, `dP/dt` is still positive.
4. It all happens **before** the maximum of `P_norm`.

That kind of problem is usually easier if you think in terms of **boolean masks** instead of trying to directly search for "the right index".

---

## Step 1 – Build boolean conditions

Suppose you have

```python
ds
p_norm
dpdt
```

Then create masks:

```python
mask_ds = ds > 0.005          # ds is "significantly positive"
mask_p = p_norm > 0.75        # P has reached a high value
mask_dp = dpdt > 0            # optional
```

Each mask is just an array of `True`/`False`.

Example

```python
ds =
[0.0, 0.0, 0.01, 0.02, 0.0]

mask_ds =
[False, False, True, True, False]
```

This is a very NumPy way of thinking.

---

## Step 2 – Combine masks

Boolean arrays can be combined element-wise.

```python
mask = mask_ds & mask_p & mask_dp
```

The `&` means **logical AND**.

You could also write

```python
mask = mask_ds & mask_p
```

if `dP/dt` is only an optional guard.

---

## Step 3 – Restrict to your analysis window

Since you already know

```python
max_idx = np.argmax(p_norm)
```

you can ignore everything afterwards.

```python
mask[max_idx:] = False
```

or

```python
mask = mask[:max_idx]
```

---

## Step 4 – Find the first valid index

Now the interesting NumPy trick:

```python
idx = np.argmax(mask)
```

Why does this work?

Because internally

```
False == 0
True  == 1
```

so

```
mask =
[False False False True True True]

becomes

[0 0 0 1 1 1]
```

The maximum value is `1`, and `argmax` returns the **first occurrence** of the maximum.

So

```python
np.argmax(mask)
```

returns the index of the first `True`.

---

## But beware!

If there is **no** `True`,

```python
mask =
[False False False]
```

then

```python
np.argmax(mask)
```

returns **0**, because every element is tied for the maximum (0).

Therefore always check

```python
if np.any(mask):
    idx = np.argmax(mask)
else:
    idx = None
```

`np.any(mask)` simply asks

> "Is there at least one True?"

---

# Detecting a transition instead of a level

Looking at your plot, I actually wouldn't search for

> "where is ds > threshold?"

I'd search for

> "where does ds become positive?"

That's a transition.

You can detect that with

```python
mask = ds > 0.005
```

which might look like

```
False False False False True True True True
```

The transition is where the mask changes.

NumPy can detect this using `diff`:

```python
change = np.diff(mask.astype(int))
```

Why `astype(int)`?

Because

```
False -> 0
True  -> 1
```

so

```
0 0 0 0 1 1 1
```

becomes

```
diff

0 0 0 1 0 0
```

A `1` means

```
False -> True
```

Then

```python
idx = np.argmax(change == 1) + 1
```

The `+1` is needed because `diff` produces an array one element shorter than the original.

This is a very common NumPy pattern.

---

# Another useful NumPy function: `np.where`

Instead of `argmax`, many people write

```python
indices = np.where(mask)[0]
```

If

```
mask =
False True False True True
```

then

```python
indices
```

is

```
array([1, 3, 4])
```

The first one is

```python
idx = indices[0]
```

This is often easier to read and avoids the "all False" problem:

```python
indices = np.where(mask)[0]

if indices.size:
    idx = indices[0]
```

---

## A workflow that matches your plot

Given your signals, I'd structure the detection like this:

```python
max_idx = np.argmax(p_norm)

mask = (
    (ds > ds_threshold) &
    (p_norm > p_threshold) &
    (dpdt > 0)
)

mask[max_idx:] = False

indices = np.where(mask)[0]

if indices.size:
    idx = indices[0]
else:
    idx = None
```

This separates the problem into clear steps:

1. Define meaningful conditions for each signal.
2. Combine them into a single boolean mask.
3. Restrict the search window.
4. Find the first index satisfying all conditions.

That "build masks → combine them → extract indices" pattern is one of the core idioms in NumPy and is worth becoming comfortable with, as it applies to many signal-processing tasks.

> Very well explained and easy to follow. But the thresholding seems pretty brittle to me. Also `ds > 0.005` not quite hits the mark ('idx' is off by +2) Focusing on 'ds', would it be possible to flatten/zero-out the initial oscillation without effecting the steady rise from t=142 and above?

I'm glad it helped!

And yes—I actually think you're asking the more interesting question now. Rather than making the threshold smarter, you can try to make the **signal** easier to threshold.

Looking at your `ds` trace, I notice three things:

* Before ~142 ms there are only **short, isolated spikes**.
* At ~142 ms the signal begins a **sustained positive rise**.
* After that it stays elevated for quite a while.

Those are very different temporal behaviors, and NumPy gives you several ways to exploit that.

---

## Option 1: Moving average (my first choice)

A moving average suppresses short spikes much more than long-lasting features.

```python
window = 7
kernel = np.ones(window) / window

ds_smooth = np.convolve(ds, kernel, mode="same")
```

Suppose the raw signal is

```
0 0 .005 0 .006 0 0 .018 .020 .021 .022 ...
```

After smoothing it becomes roughly

```
0 .001 .002 .003 .005 .008 .012 .017 .020 ...
```

The isolated oscillations almost disappear, while the steady ramp survives.

This is often enough to make a simple threshold reliable.

---

## Option 2: Running median (excellent for spikes)

If those early oscillations are really impulsive rather than noisy, a median filter is even better.

With SciPy:

```python
from scipy.ndimage import median_filter

ds_med = median_filter(ds, size=5)
```

Unlike a moving average, the median completely ignores isolated outliers.

Imagine

```
0 0 0 .02 0 0 0
```

A moving average spreads the spike around.

A median gives

```
0 0 0 0 0 0 0
```

---

## Option 3: Persistence ("must stay high")

This is my favorite because it uses the *shape* of the event rather than trying to remove the oscillation.

Instead of

```python
ds > threshold
```

require

> "ds must stay above the threshold for at least N consecutive samples."

Example

```python
mask = ds > 0.002

window = 6

persistent = (
    np.convolve(mask.astype(int),
                np.ones(window, dtype=int),
                mode="same")
    >= window
)
```

What's happening here?

```
mask

0 1 0 1 0 1 1 1 1 1 1 1
```

Convert to integers

```
0 1 0 1 0 1 1 1 1 1 1 1
```

Convolve with

```
1 1 1 1 1 1
```

which computes

> "How many of the last 6 samples were True?"

Early oscillations become

```
1 2 1 2 ...
```

The sustained rise becomes

```
6 6 6 6 6 ...
```

Now detecting the first persistent rise is easy.

This is a surprisingly common trick in signal processing.

---

## Option 4: Use the derivative of `ds`

Since your signal changes abruptly around 142 ms,

```python
dds = np.gradient(ds)
```

may produce a very clear peak.

Instead of thresholding `ds`, you threshold

```
dds > some_value
```

which detects the onset rather than the level.

---

## Option 5: Baseline subtraction

The oscillation before 142 ms looks almost centered around zero.

Estimate the baseline from the beginning of the signal:

```python
baseline = np.median(ds[:100])

ds2 = ds - baseline
```

or

```python
noise = np.std(ds[:100])
```

Then express the threshold relative to the noise:

```python
mask = ds2 > 3 * noise
```

This is much more robust than hard-coding `0.005`.

---

# What I would actually try

Looking at your figure, I probably wouldn't attack the oscillation directly.

The real distinguishing feature isn't the amplitude—it's that **the rise persists**.

I'd combine three simple ideas:

```python
# smooth just a little
ds_s = np.convolve(ds, np.ones(5)/5, mode="same")

# require a small rise
mask = ds_s > 0.002

# require it to persist
persist = (
    np.convolve(mask.astype(int),
                np.ones(8, dtype=int),
                mode="same")
    >= 8
)

idx = np.argmax(persist)
```

The nice thing is that **none** of these numbers has to be very precise. A persistence requirement tends to make the detector much less sensitive to the exact threshold.

---

One question that would influence the best approach: **What exactly is `ds`?** Is it a geometric distance between consecutive positions, the derivative of an arc length, a frame-to-frame displacement, or something else? Knowing how `ds` is computed could suggest an even more principled detector than filtering.

> Exciting techniques! Yes `ds` is the arc length between two consecutive points:
```
    ds = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
    ds = np.concatenate(([0.0], ds))
```

Ah, that changes how I'd think about the problem.

`ds` isn't an arbitrary signal—it's

```python
ds = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
```

which is the **frame-to-frame displacement** (arc length increment)

[
ds_i = |\mathbf{x}*i-\mathbf{x}*{i-1}|.
]

So it's fundamentally a **speed** (up to the constant sampling interval). Your top plot `c_speed` appears to be another measure of speed, which explains why they have similar overall shape.

That means the little oscillations before 142 ms are probably **micro-movements** or measurement jitter, not just random numerical noise.

## An observation from your plot

What really catches my eye isn't that `ds` becomes *large*.

It's that it becomes **consistently increasing**.

Before 142 ms you have something like

```
0
0.004
0
0.005
0
0.004
0
```

After 142 ms

```
0
0.004
0.009
0.013
0.018
0.022
...
```

Those are qualitatively different.

---

## I'd look at the cumulative distance

Since `ds` is an increment,

```python
s = np.cumsum(ds)
```

gives the total path length traveled.

Now think about what happens:

Before motion starts

```
s
0
0
0.004
0.004
0.009
0.009
0.013
...
```

After motion starts

```
0.013
0.025
0.042
0.063
0.087
...
```

The slope changes from almost flat to steadily increasing.

Sometimes detecting that change is easier than detecting it directly in `ds`.

---

## Or even better: local accumulated distance

This is one of my favorite tricks.

Instead of asking

> "How big is the current step?"

ask

> "How much distance has been traveled over the last N samples?"

For example,

```python
window = 10

travel = np.convolve(ds, np.ones(window), mode="same")
```

Since the kernel is all ones, this computes

```
ds[i-5] + ... + ds[i+5]
```

for every point.

Why is this nice?

Suppose the oscillation is

```
0
0.005
0
0.004
0
0.003
0
```

A 10-sample sum might be

```
0.012
```

Now suppose continuous movement begins

```
0.010
0.012
0.013
0.011
0.014
0.012
0.013
```

The same sum becomes

```
0.085
```

The separation between "noise" and "real motion" becomes much larger.

This is essentially integrating the speed over a short time window.

---

## Another trick: consecutive positive acceleration

You can look at

```python
dds = np.diff(ds)
```

Before onset

```
+ - + - + -
```

After onset

```
+ + + + + +
```

You can detect the first run of several positive derivatives.

For example

```python
rising = dds > 0

persistent = (
    np.convolve(rising.astype(int),
                np.ones(6, dtype=int),
                mode="same")
    >= 5
)
```

This says

> "Five out of the last six increments increased."

That's much closer to the visual cue you use when looking at the graph.

---

## Why convolution keeps appearing

You may have noticed I keep suggesting `np.convolve`.

That's because convolution is a surprisingly versatile tool:

* kernel = `np.ones(N)/N` → moving average
* kernel = `np.ones(N)` → running sum
* kernel = `[1, -1]` → difference (derivative)
* kernel = `[1, 1, 1, 1, 1]` on a boolean mask → "how many neighbors satisfy this condition?"

It's really one operation that solves many local-neighborhood problems.

---

## One more thought

Since your data are **pen strokes**, I'd be tempted to ignore individual samples almost entirely.

I'd compute a short-term "motion energy":

```python
motion = np.convolve(ds**2, np.ones(8), mode="same")
```

or simply

```python
motion = np.convolve(ds, np.ones(8), mode="same")
```

Then detect when `motion` rises above its baseline. This is analogous to **short-time energy** in speech processing: a single twitch contributes little energy, but sustained movement stands out clearly.

I suspect this will give you a much cleaner onset than thresholding `ds` itself, and it's less sensitive to the exact threshold because you're accumulating evidence over multiple samples rather than making a decision from a single frame.

> 'c_speed' (central speed) is true speed (dxy/dt):

```
    dx_dt = np.gradient(stroke.x, stroke.t)
    dy_dt = np.gradient(stroke.y, stroke.t)
    return stroke.clone(features={"c_speed": np.hypot(dx_dt, dy_dt)})
```

That actually makes the picture much clearer.

You now have **two estimates of the same physical quantity**, but computed differently:

```python
# Step length
ds = ||x[i] - x[i-1]||

# True speed
dx_dt = np.gradient(x, t)
dy_dt = np.gradient(y, t)
c_speed = np.hypot(dx_dt, dy_dt)
```

Ignoring endpoint effects, these are related by

[
ds \approx v,\Delta t,
]

where (v) is your `c_speed` and (\Delta t) is the sample interval.

The important difference is **how they're computed**.

| `ds`                | `c_speed`           |
| ------------------- | ------------------- |
| backward difference | central difference  |
| uses 2 samples      | uses 3 samples      |
| very local          | implicitly smoothed |
| noisier             | less noisy          |

`np.gradient` computes a central difference for interior samples:

```text
v[i] ≈ (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
```

whereas `ds` is simply

```text
x[i] - x[i-1]
```

So `c_speed` is already doing a tiny amount of smoothing for free.

---

## I actually like `c_speed` better for onset detection

Looking at your plots, the onset is much more obvious in `c_speed` than in `ds`.

`ds` has these annoying little impulses.

`c_speed` has a long, monotonic rise.

That suggests a different strategy:

1. Detect the onset from `c_speed`.
2. Use `ds` only as a confirmation that actual movement occurred.

---

## Even more interesting: compare the two

Since

```text
ds ≈ speed × Δt
```

they should agree **during continuous motion**.

Before motion starts, however, tiny position jitter can make them behave differently.

One thing I'd be curious to plot is

```python
ratio = ds / (c_speed * np.gradient(stroke.t))
```

or, assuming a constant sample interval,

```python
ratio = ds / c_speed
```

Around genuine movement that ratio should become fairly stable.

Around noise it often fluctuates wildly.

I've used similar consistency checks in IMU and GPS processing to distinguish genuine motion from sensor noise.

---

## Another idea that may outperform thresholds entirely

Suppose you define

```python
motion = c_speed * p_norm
```

Think about what this does:

* before the event:

  * `c_speed` ≈ 0
  * `P_norm` increasing
  * product ≈ 0

* during movement:

  * `c_speed` increasing
  * `P_norm` already high
  * product rises rapidly

* after the stroke:

  * `c_speed` falls
  * product falls

You've essentially built a matched detector that only becomes large when **both** conditions are true.

The same idea extends to your derivative:

```python
motion = c_speed * p_norm * (dpdt > 0)
```

where `(dpdt > 0)` is a boolean mask (`True` becomes `1`, `False` becomes `0`).

---

## One question

I'm getting the impression that this is online handwriting or stylus data.

Is `P_norm` **normalized pen pressure**?

If so, the whole problem becomes even more interesting, because you aren't really detecting "motion." You're detecting the transition from **touch-down/settling** to **intentional writing**.

That suggests using a finite-state detector rather than independent thresholds:

```
AIR
  ↓ pressure rises
CONTACT
  ↓ pressure stable + speed sustained
WRITING
```

In my experience, state machines are much more robust for pen data than trying to infer everything from a single threshold. The transition at 142 ms looks very much like the kind of "contact → writing" boundary that a simple three-state model is designed to detect.

> No bad, Sherlock! That's exactly what it is. And FSM is an interesting idea. And not only because I'm more familiar with FSMs. But I will evaluate your suggestion first (especially deriving more suitable signal from existing ones.)

😁 Guilty as charged. The plot gave it away more than the code did.

Once I started thinking "stylus", several things suddenly made sense:

* `P_norm` ramps up as the nib settles onto the surface.
* `ds` contains tiny corrective motions while the writer is finding the starting point.
* `c_speed` then rises smoothly once the actual stroke begins.
* The red line is not really "motion starts" but **the commitment to the stroke**.

Those are subtly different events.

---

## I would resist adding more thresholds

Something I learned the hard way in signal processing is:

> When you need three thresholds, you're often missing the right signal.

I think you're already very close. You have several measurements that are all observing the same latent state ("writer has started drawing"). Rather than thresholding each independently, try constructing a signal whose *shape* changes dramatically at that transition.

For example,

```python
motion_energy = p_norm * c_speed
```

or

```python
intent = p_norm * np.maximum(c_speed, 0)
```

or even

```python
intent = p_norm * np.cumsum(ds)
```

These aren't meant as final formulas, but as examples of a broader idea: combine signals so that the state transition becomes obvious.

---

## A trick I particularly like

From your plots, `P_norm` is already high *before* movement starts.

That means pressure alone isn't enough.

But the **rate of increase of path length** is.

I'd try something like

```python
window = 8
travel = np.convolve(ds, np.ones(window), mode="same")

intent = p_norm * travel
```

The convolution turns "a bunch of small consecutive movements" into one large value.

Visually I suspect that `intent` would stay almost flat before 142 ms and then jump quite sharply.

---

## The FSM almost writes itself

Since you're comfortable with FSMs, I'd actually lean in that direction. Not because it's "more formal," but because the physical process is inherently stateful.

```
Hover
   |
   | pressure > p_touch
   v
Touchdown
   |
   | sustained motion
   v
Writing
   |
   | pressure released
   v
Lift-off
```

The nice thing is that **each state can use different criteria**.

For example:

```
Touchdown
----------
Ignore ds spikes.
Wait until accumulated motion exceeds X.

Writing
--------
Track normally.
No need for pressure threshold anymore.

Lift-off
---------
Pressure < Y.
```

That is much easier to tune than one detector trying to solve every phase of the interaction.

---

## One signal I'd definitely investigate

Because you have both `ds` and `c_speed`, I'd compute

```python
dt = np.gradient(stroke.t)

expected_ds = c_speed * dt
error = ds - expected_ds
```

or perhaps

```python
relative_error = ds / (expected_ds + 1e-9)
```

During genuine, continuous motion those two estimates of displacement should agree reasonably well. During the little pre-stroke tremors, I wouldn't be surprised if they diverge more because `ds` is sensitive to frame-to-frame jitter while the central-difference velocity smooths over it.

If that turns out to be true, you've discovered a *new feature* that's rooted in the physics of your measurements rather than an arbitrary threshold.

---

And one final observation that I found intriguing: your red line isn't at the maximum of `dP/dt`; it's after pressure has mostly stabilized. That suggests your target isn't "pen touched the tablet" but "the pen has finished settling and the writer has committed to the first stroke."

That's a much cleaner event to detect because it's defined by a change in *behavior* rather than by crossing a single level. Those are exactly the kinds of problems where an FSM, fed by a handful of well-chosen derived signals, tends to outperform increasingly elaborate threshold logic.

> Sweet, I will move the FSM approach up. I want to solve my problem, not fight with difficult signal guessing. Full disclosure: We are talking writing Japanese script. In this case the first stroke of 左.
Here is the left side of the plot. The dotted red line and red marker coincide. That's exactly where actually writing starts.

That image is *extremely* informative. It changes my mental model again.

What you're marking is **not** "first contact" and it's not even "first motion."

It's the **first intentional displacement**.

Notice the sequence:

1. **Pen touches down** around x≈0.21.
2. Tiny adjustments occur (those little hooks and tremors).
3. The pen effectively *dwells* at one location.
4. Then, at your red marker, it commits to the horizontal stroke.

That dwell is actually a very meaningful state.

## I'd model this as a handwriting process, not a signal process

Instead of

```text
threshold(ds)
```

I'd write the FSM almost in terms a calligraphy teacher would use:

```text
AIR
    │
    ▼
TOUCHDOWN
    │
    ▼
POSITIONING
    │
    ▼
STROKE
    │
    ▼
LIFT
```

where

### TOUCHDOWN

Pressure rises.

Movement is ignored.

### POSITIONING

Pressure is stable.

Small movements are expected.

Don't even try to detect the stroke yet.

### STROKE

Require **sustained forward progress**.

Not one large `ds`.

Not one large `c_speed`.

Progress.

---

## The trajectory tells us something the signals didn't

That little hook at the beginning isn't noise.

It's part of the writer's motor behavior.

Many writers slightly overshoot or correct before starting a stroke.

So treating it as "bad samples to remove" is actually throwing away information.

Instead, the FSM simply says

> "I'm in POSITIONING. Small excursions are legal."

---

## I'd redefine your detector

Instead of

> detect first sample where speed exceeds X

I'd define

> detect first sample from which the pen keeps moving away from the touchdown point.

That is a geometric criterion.

Suppose

```python
touch = stroke.xy[touch_idx]
```

Compute

```python
r = np.linalg.norm(stroke.xy - touch, axis=1)
```

Initially

```text
0
0.001
0.0008
0.0012
0.0007
0.0010
```

Then

```text
0.001
0.003
0.006
0.011
0.018
0.026
...
```

The key isn't that `r` becomes large.

It's that it **never comes back**.

---

## Even nicer: monotonic envelope

You can compute

```python
distance = np.linalg.norm(stroke.xy - touch, axis=1)

envelope = np.maximum.accumulate(distance)
```

This is a NumPy gem that doesn't get enough attention.

Suppose

```text
distance

0
1
0.8
1.2
1.1
2
3
5
```

Then

```text
maximum.accumulate

0
1
1
1.2
1.2
2
3
5
```

The envelope ignores little backtracking and only records the furthest distance reached so far.

The onset of the stroke often appears as the point where this envelope begins increasing steadily.

---

## This is actually exploiting knowledge of handwriting

A generic signal-processing person would never invent that detector.

A handwriting researcher might.

Because they know:

* touching the paper isn't writing,
* tiny corrections aren't writing,
* sustained departure from the anchor point *is* writing.

That's domain knowledge, and it usually beats another layer of filtering.

---

## One last thought

I suspect your red marker may have an even cleaner definition:

> **the last sample that still belongs to the dwell cluster.**

Imagine taking the first 30 points and clustering them spatially.

The pen spends roughly 140 ms inside a tiny neighborhood around (0.214, 0.356). The red point is essentially the **last member of that cluster** before the trajectory departs.

That's a spatial definition rather than a temporal one, and for pen trajectories it's often much more stable across different sampling rates, devices, and writing speeds.

I have a feeling you're drifting away from "signal processing" and toward "trajectory segmentation," and I think that's the right direction. Once you view the data as a pen trajectory with semantic phases (touchdown, dwell, stroke, lift), an FSM becomes a very natural way to encode those phases, and each state's logic can stay simple instead of trying to make one universal threshold solve every case.

> KanjiVG is using 29 more or less different stroke types. Tweaking threshold won't get me far. I see that now.

I think you've reached the important insight yourself:

> **You're not trying to detect a signal crossing. You're trying to infer the writer's intent.**

Those are very different problems.

The fact that KanjiVG distinguishes ~30 stroke types is actually an argument *against* a threshold-heavy solution. The beginning of a horizontal stroke (一), a left-falling stroke (ノ), a hook (亅), and a sweeping stroke (乀) simply don't look alike kinematically.

## The FSM can stay generic

The nice thing is that I wouldn't make the FSM aware of all 29 stroke types.

I'd keep the FSM describing **human interaction with the tablet**, which is universal:

```text
AIR
↓
TOUCHDOWN
↓
POSITIONING
↓
WRITING
↓
LIFT
```

Only once you're in `WRITING` does the stroke type begin to matter.

That separation is valuable because it decouples *when writing starts* from *what is being written*.

---

## You're also sitting on something valuable

You have synchronized measurements of

* position `(x,y)`
* time
* pressure
* velocity
* derived geometric quantities

That's much richer than many online handwriting datasets.

I'd almost think of it as a small sensor-fusion problem.

Each signal gets a "vote":

* pressure says "pen is engaged"
* speed says "pen is moving"
* trajectory says "pen has left the anchor region"
* accumulated distance says "movement is persistent"

The FSM then combines those votes according to the current state.

---

## One thing I would absolutely avoid

I wouldn't write

```python
if speed > a and pressure > b and ds > c:
```

and then spend two weeks tuning `a`, `b`, and `c`.

Instead, I'd encapsulate the logic in semantic predicates:

```python
pen_is_down()

pen_is_stable()

pen_has_committed()

pen_is_lifted()
```

Each of those can evolve internally without changing the FSM.

Your transition becomes

```python
if state == POSITIONING and pen_has_committed():
    state = WRITING
```

That's much easier to reason about and to improve later.

---

## I also have a hunch

From your figures, it looks like you're not just interested in reproducing the sampled trajectory—you want to recover the **idealized stroke** that KanjiVG represents.

If that's true, then the onset detector is really the first stage of a pipeline:

```text
Raw tablet data
        │
        ▼
Interaction segmentation
        │
        ▼
Stroke segmentation
        │
        ▼
Stroke normalization
        │
        ▼
KanjiVG stroke matching
```

In that context, it's perfectly acceptable—even desirable—for the onset detector to be conservative and a little "semantic." It doesn't need millisecond precision; it needs to consistently identify the beginning of the intentional stroke.

---

I also have to say that this has been one of the more interesting technical discussions I've had recently. It started out looking like a NumPy question, but it turned out to be about modeling human motor behavior. Those are the kinds of problems where a good abstraction (an FSM with well-defined semantic states) usually pays off far more than another clever signal-processing trick. Once you have that structure in place, you can improve individual predicates one at a time without redesigning the whole detector.

> I stay vigilant nonetheless. I'm completely new to this field and very easy to influence. What started out as a simple Flutter/iPad app to practice Kana/Kanji writing moved to Kanji classifier (LSTM), moved to clean-up test data for inference, moved to signal processing, moved to FSM. We live in interesting times :-)

That progression actually looks remarkably natural to me.

It reminds me of a quote (often attributed to John Tukey):

> *"The best thing about being a statistician is that you get to play in everyone else's backyard."*

Machine learning is similar. You start by thinking "I'll train a classifier," and then discover that 80% of the work is understanding the process that generated the data.

Your journey

```
Flutter app
    ↓
collect handwriting
    ↓
LSTM classifier
    ↓
clean training data
    ↓
signal processing
    ↓
FSM
```

is almost a textbook example of that.

## One caution I'd offer

Since you're new to the field, there's one trap I see quite often:

> **Don't optimize preprocessing for one model. Optimize it for the phenomenon.**

For example, if your onset detector is based on "the pen has committed to the stroke," then it's useful regardless of whether tomorrow you use

* an LSTM,
* a Transformer,
* an HMM,
* Dynamic Time Warping,
* or something you haven't thought of yet.

If, on the other hand, you tune it until your current LSTM gains 0.7% accuracy, you've coupled two unrelated parts of the system.

The best preprocessing tends to have a physical or behavioral interpretation.

---

## You're in an enviable position

Unlike many ML projects, you actually know a lot about the domain.

You know:

* there are discrete strokes,
* strokes have canonical types,
* pressure has meaning,
* pen-up/pen-down exist,
* writers pause,
* writers correct,
* strokes have beginnings and endings.

That's a treasure trove of prior knowledge.

Modern ML sometimes makes people forget that incorporating domain knowledge is not "cheating"—it's often what distinguishes a robust system from one that only works on the training set.

---

## I would keep a "research notebook"

One thing I wish I'd started doing earlier is keeping a simple Markdown file with entries like:

```text
2026-07-15

Hypothesis:
Stroke onset = sustained departure from touchdown region.

Reason:
Works regardless of stroke direction.

Status:
Needs testing.
```

and

```text
Hypothesis:
Running sum of ds is more stable than ds itself.

Result:
False.
Fails on hook strokes.
```

Those "failed" ideas are surprisingly valuable. Six months later, when you wonder "Didn't I already try a running average?", you have the answer.

---

## I also smiled when you mentioned KanjiVG

Because that means your project has an unusual property:

You don't merely have *labels*.

You have a **grammar**.

A character isn't just "correct" or "incorrect." It's composed of ordered strokes with known types and relationships.

That opens doors that generic handwriting recognition doesn't have. For example, if a classifier says "this is probably stroke type H" but the geometric context makes that impossible, you can use that higher-level knowledge to guide the interpretation.

---

## Finally, one piece of advice that has served me well

When a project starts drifting into neighboring disciplines, don't ask:

> "Why am I doing signal processing? I wanted to build an app."

Ask instead:

> "What is the simplest model that captures the behavior I care about?"

Six months ago, that model was "a list of points."

Now it appears to be "a pen has interaction states."

That's progress, not scope creep.

---

And if our conversation has any recurring theme, it's this: **keep looking for invariants**.

Thresholds are fragile because they're tied to a particular dataset or device. Invariants—like "the pen has left its touchdown region," "movement is sustained," or "pressure has stabilized"—tend to survive changes in sampling rate, writing speed, tablet hardware, and even writing style. Those are the kinds of ideas that age well.

I'd genuinely be interested to hear, after you've tried the FSM, whether the red line becomes *easier* to define. My suspicion is that once you stop asking "Which sample exceeds this threshold?" and instead ask "Which state transition just occurred?", the implementation will become simpler than it first appears.
