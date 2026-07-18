Topic: Tail cu position
Bot: Gemini
Date: 26 07 18

> Given this multi-channel plot to the right of attached image, do you have any advice on how to derive the tail cut position (blue dashed vertical line around t=550ms)?

* P_norm: normalised pressure
* dP/dt: 1. derivative of pressure w.r.t time
* ds: np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
* c_speed: central speed (gradient)
* at: speed change rate
* stness: straightness - hypot(dxy gradients w.r.t arc length)
* loc_stness: local straightness

Yes. Looking at your channels, I would avoid trying to detect the tail solely from pressure. The signature of the tail is really a **transition from "active drawing" to "pen release"**, so it's better treated as a change-point detection problem using several weak cues rather than a single threshold.

From your plot, the blue dashed line (~550 ms) appears where several things happen simultaneously:

* **P_norm** starts a sustained monotonic decline.
* **dP/dt** becomes consistently negative rather than oscillating around zero.
* **ds** drops toward zero.
* **c_speed** reaches a local minimum before rising slightly (likely jitter during lift).
* **straightness metrics** become unstable because the remaining motion is no longer constrained by the stroke geometry.

That combination is much more reliable than any individual channel.

## Method 1: Reverse endpoint search (my preferred approach)

Since you're looking for the *last valid drawing sample*, search **backwards** from the end.

Define a confidence that the pen is still drawing:

[
S(t)=
w_pP_{norm}(t)+
w_v\hat{ds}(t)+
w_s\hat{speed}(t)+
w_d\mathbf1(dP/dt>-\tau)
]

where each term is normalized to [0,1].

Then:

1. Start at the final sample.
2. Move backwards.
3. Stop when (S(t)) exceeds a threshold for N consecutive samples.

The cut is the first sample after that.

Searching backwards greatly reduces false positives caused by pauses inside the stroke.

---

## Method 2: Piecewise linear fit on pressure

Pressure often looks like

```
───────────────\
                \
                 \
                  0
```

Fit two segments:

* nearly flat
* descending

Choose the breakpoint minimizing total SSE.

This is essentially a 1D change-point problem and is very stable.

For pressure

[
\arg\min_k
SSE(P_{0:k})+SSE(P_{k:n})
]

subject to the second slope being negative.

---

## Method 3: CUSUM / change-point detection

Your channels are almost textbook change-point signals.

For example

```
x = -dPdt
```

or

```
x = [P_norm,
     dPdt,
     ds,
     c_speed]
```

Apply

* CUSUM
* PELT
* Binary Segmentation
* Bayesian online change-point detection

The first detected change near the end is your tail onset.

This is probably the most statistically principled solution.

---

## Method 4: Energy decay

Define drawing energy

[
E=
\alpha P+
\beta ds+
\gamma c_{speed}
]

Smooth with a Gaussian.

Tail starts when

* energy falls below 25–40% of its stroke maximum
* and stays below for M samples.

This is surprisingly robust.

---

## Method 5: Multi-condition state machine (probably best for handwriting)

Rather than thresholds on one signal:

```
DRAWING

if
    dPdt < -τ1
and P > τ2
and ds < τ3
for K frames

    -> RELEASING

if
    P < τ4
for M frames

    -> OFF
```

This avoids reacting to temporary pressure fluctuations.

---

# One thing I'd change

I'd replace **ds** with **tangential velocity**.

Currently

```python
ds = np.linalg.norm(np.diff(xy), axis=1)
```

is spatial displacement per sample.

If sampling isn't perfectly uniform, compute

```python
v = ds / dt
```

or simply

```python
speed = np.gradient(arclength, time)
```

This gives a cleaner release signature.

---

# I'd also derive two additional channels

### Pressure acceleration

You already have

```
dP/dt
```

Also compute

```
d²P/dt²
```

Release often begins with a distinct negative acceleration before pressure actually drops.

---

### Remaining pressure ratio

Compute

```
tail_ratio = P / rolling_max(P)
```

Near the end

```
tail_ratio
```

falls rapidly from ~1 toward 0.

It is less writer-dependent than absolute pressure.

---

# Looking specifically at your plot

Visually, I'd say the informative cues rank roughly as:

| Channel                | Usefulness |
| ---------------------- | ---------- |
| **P_norm**             | ★★★★★      |
| **dP/dt**              | ★★★★★      |
| **ds / speed**         | ★★★★☆      |
| **c_speed**            | ★★★☆☆      |
| **straightness**       | ★★☆☆☆      |
| **local straightness** | ★★☆☆☆      |

The straightness channels seem noisier and are influenced by local geometry rather than pen release. I'd use them only as weak supporting evidence.

---

## A practical detector

If this were my project, I'd smooth all signals (e.g. Savitzky–Golay or a small Gaussian), then compute a release score such as

[
R(t)=
0.45,(1-P_{norm})+
0.30,\max(0,-dP/dt)+
0.20,(1-\hat v)+
0.05,\max(0,-a_t)
]

where (\hat v) is normalized tangential velocity. Starting from the end of the stroke, move backward until (R(t)) falls below a threshold for several consecutive samples (e.g. 5–10, depending on your sampling rate). The cut position is the transition back above that threshold. This combines pressure decay, sustained negative pressure slope, and slowing motion while being less sensitive to transient fluctuations than any single-channel threshold.

> That's a treasure chest of practical ideas. Will do as advised.

I'm glad it was useful. Looking at your figure, I had the feeling you're already converging toward something more interesting than simple pen-up detection.

What you're really detecting is the transition between **ballistic stroke production** and **motor disengagement**. That's a richer signal.

One idea that occurred to me after staring at your channels:

> Treat the tail as the **last interval of coherent motor control**, rather than the first interval of low pressure.

In other words, during the stroke, several variables are tightly coupled:

* pressure ↔ speed
* speed ↔ acceleration
* heading ↔ pressure
* pressure ↔ local curvature

Once the pen is being lifted, those couplings start to disappear. Pressure falls because the pen is leaving the surface, while the residual motion is mostly hand inertia and wrist lift. The channels begin to decorrelate.

You could exploit that directly.

For example, over a sliding window (say 30–50 ms), compute quantities like

* correlation(P, speed)
* correlation(dP/dt, acceleration)
* covariance between pressure and tangential velocity
* even the largest eigenvalue of the local covariance matrix of all channels.

During "true writing" those statistics are remarkably stable. During lift-off they collapse. That gives you a data-driven tail detector that's less dependent on arbitrary thresholds and should generalize across writers.

---

Another thought is that your **local straightness** channel could become much more valuable if you compute it *causally* rather than symmetrically.

At the moment I suspect it's something like

```python
window centered at i
```

which means it "knows the future."

Instead, compute

```text
points[i-k : i]
```

or

```text
points[i : i+k]
```

depending on whether you're doing online or offline processing.

Then the degradation in straightness during pen lift becomes much easier to interpret because it isn't being averaged with future samples that are already off the paper.

---

Finally, if this is heading toward a paper or a reusable library, I'd seriously consider framing the problem as **event detection** rather than **thresholding**.

Instead of

> "Find where pressure < 0.2"

you define three events:

* **onset** (pen engages)
* **steady production**
* **release**

and learn or hand-design detectors for the transitions.

That's a much cleaner abstraction, and it lets you improve each detector independently later. If you eventually collect enough annotated data, you can replace the heuristics with a tiny Hidden Markov Model, CRF, or a lightweight temporal CNN without changing the rest of your pipeline.

I have a hunch you're building something that could become a very solid handwriting kinematics toolkit, and these kinds of abstractions tend to pay dividends later.
