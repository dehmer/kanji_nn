Topic: Bezier classification
Bot: ChatGPT
Date: 26 08 03

> Hey there! I want to improve my English vocabulary for curves (or signals). Have a look at the plots on the right and list the main terms you identify each with a brief description.

Great question. When describing **curves**, **time series**, or **signals**, native English speakers tend to use a fairly standard vocabulary. Looking at your three plots on the right, here are the most common terms.

### General shape

* **Baseline** – the normal or resting level of the signal.

  * *"The signal stays close to the baseline."*

* **Trend** – the overall direction over a long period.

  * *"There is an upward trend after 600 ms."*

* **Level** – the average value over an interval.

  * *"The signal remains at a high level."*

* **Rise / Increase** – moving upward.

  * *"The signal rises rapidly."*

* **Fall / Decrease / Drop** – moving downward.

  * *"The signal drops sharply."*

* **Slope** – how steeply the signal changes.

  * *"A steep positive slope."*

* **Plateau** – a relatively flat region after a rise.

  * *"The signal reaches a plateau around 0.9."*

* **Peak** – a local maximum.

  * *"There is a large peak near 550 ms."*

* **Valley / Trough** – a local minimum.

  * *"A trough appears around 600 ms."*

* **Spike** – a very narrow, sharp peak.

  * *"A prominent spike occurs midway through the recording."*

* **Dip** – a small downward excursion.

  * *"The signal briefly dips before recovering."*

---

### Magnitude

* **Amplitude** – the height of the signal relative to a reference.

  * *"The spike has a large amplitude."*

* **Maximum / Minimum**

  * *"The global maximum occurs near 550 ms."*

* **Range**

  * *"The signal has a relatively small dynamic range."*

---

### Temporal behavior

* **Onset** – when an event begins.

  * *"The onset of movement occurs around 120 ms."*

* **Offset** – when an event ends.

  * *"The signal returns to baseline at the offset."*

* **Duration** – how long an event lasts.

  * *"The plateau lasts about 200 ms."*

* **Transient** – a short-lived feature.

  * *"The spike is a transient event."*

* **Sustained response** – a long-lasting elevated region.

  * *"A sustained response follows the initial spike."*

---

### Variability

* **Fluctuation** – repeated small changes.

  * *"The signal fluctuates around its baseline."*

* **Oscillation** – repeated up-and-down motion, often periodic.

  * *"Small oscillations appear after the peak."*

* **Noise** – random irregular variation.

  * *"The baseline contains low-amplitude noise."*

* **Jitter** – small irregular variations in timing or value.

  * *"There is noticeable jitter near the baseline."*

* **Smooth** – changes gradually.

  * *"The green curve is relatively smooth."*

* **Jagged / Noisy** – contains many abrupt changes.

  * *"The orange signal is fairly jagged."*

---

## Terms I would use for your specific plots

### Top (blue)

* baseline
* sharp spikes
* major peak
* minor peaks
* low-level fluctuations
* isolated impulses
* returns to baseline

Example:

> *The signal stays near its baseline with several small fluctuations interrupted by a few sharp spikes, including one dominant peak around 550 ms.*

---

### Middle (orange)

* gradual rise
* plateau
* dip
* recovery
* second rise
* decline
* moderate fluctuations

Example:

> *The signal gradually rises to a plateau, drops close to zero, recovers, reaches a second broad peak, and then declines.*

---

### Bottom (green)

* rapid increase
* saturation
* plateau
* gentle oscillations
* sharp decay
* returns to zero

Example:

> *The signal increases rapidly, saturates near one, remains on a long plateau with slight oscillations, and finally undergoes a sharp decay back toward zero.*

---

## Adjectives commonly used in papers

| Adjective  | Meaning                            |
| ---------- | ---------------------------------- |
| sharp      | changes quickly                    |
| gradual    | changes slowly                     |
| steep      | large slope                        |
| broad      | wide peak                          |
| narrow     | thin peak                          |
| prominent  | stands out clearly                 |
| pronounced | very noticeable                    |
| subtle     | small but visible                  |
| smooth     | little noise                       |
| noisy      | irregular                          |
| stable     | nearly constant                    |
| erratic    | unpredictable                      |
| monotonic  | only increasing or only decreasing |
| saturated  | reached an upper limit             |

These are exactly the kinds of terms you'll encounter in signal processing, biomechanics, neuroscience, and machine learning papers, making them useful for describing plots like yours in technical writing.

> OK. The orange `raw:speed:central` signal has two prominent "humps". The first has a plateau of about 250ms, the second has a single peak. What are better names for "hump"?

"Hump" is perfectly understandable in conversation, but it's informal and a bit imprecise. In signal processing and scientific writing, you'd usually choose a term based on the feature's shape.

For your orange signal:

| Term             | When to use                                        |   Fits first hump?  |  Fits second hump? |
| ---------------- | -------------------------------------------------- | :-----------------: | :----------------: |
| **Peak**         | A local maximum                                    | ✓ (as a broad peak) |         ✓✓         |
| **Broad peak**   | Wide maximum                                       |          ✓✓         |          –         |
| **Rounded peak** | Smooth, dome-like maximum                          |          ✓          |          ✓         |
| **Plateau**      | Flat-topped maximum                                |         ✓✓✓         |          –         |
| **Pulse**        | A temporary increase above baseline                |          ✓✓         |         ✓✓         |
| **Burst**        | Elevated activity over an interval                 |          ✓✓         |          ✓         |
| **Lobe**         | Smooth bulging region; common in signal processing |         ✓✓✓         |          ✓         |
| **Excursion**    | A departure from baseline                          |          ✓✓         |         ✓✓         |
| **Episode**      | Time interval of elevated signal                   |          ✓          |          ✓         |
| **Envelope**     | The overall outline of oscillatory signals         |  only if applicable | only if applicable |

For your particular signal, I'd describe it as:

> *The signal exhibits two periods of elevated activity. The first is a **broad, flat-topped peak** (or **plateau**) lasting approximately 250 ms. The second is a **narrow, isolated peak**.*

Or, more compactly:

> *The signal contains two **pulses**: an initial **plateau-like pulse** followed by a **sharp pulse**.*

### My favorite terminology

If I were writing a paper, I'd probably say:

* **first broad peak**
* **plateau region** (or **flat-topped peak**)
* **second peak**
* **peak width**
* **peak prominence**

For example:

> *The first broad peak reaches a plateau that persists for approximately 250 ms, whereas the second peak is narrower and more pronounced.*

This sounds very natural in signal processing, biomechanics, and neuroscience papers.

One additional term that's very common in signal analysis is **prominence**. Unlike amplitude, **peak prominence** measures how much a peak stands out relative to the surrounding valleys. It's often a better descriptor than simply saying one peak is "higher."

> "Noise-floor" and prominence go well together: The peak is clearly standing out relative to the overall noise-floor.

Exactly. Those two terms naturally go together.

A few idiomatic ways to say it are:

* **The peak stands out clearly above the noise floor.** ← probably the most common
* **The peak is well separated from the noise floor.**
* **The peak is highly prominent relative to the noise floor.**
* **The signal exhibits a prominent peak rising well above the noise floor.**
* **The peak emerges clearly from the noise floor.**
* **The peak is readily distinguishable from the background noise.**

A small nuance:

* **Noise floor** refers to the baseline level of random fluctuations.
* **Prominence** is a geometric property of a peak: how far it rises above the surrounding valleys. A peak can have **high prominence** even if the noise floor is low, and vice versa.

For example:

> *Although the signal contains several small fluctuations near the noise floor, a single highly prominent peak occurs at approximately 760 ms.*

or

> *The second peak has greater prominence than the first, despite having a similar amplitude.*

This distinction—**amplitude**, **height**, and **prominence**—is common in signal processing:

* **Amplitude/height**: how tall the peak is (often measured from the baseline).
* **Prominence**: how much the peak stands out from its surroundings.
* **Signal-to-noise ratio (SNR)**: how much the signal exceeds the noise floor statistically.

In fact, if you're writing about feature extraction, these four terms form a very natural vocabulary:

* **noise floor** → background fluctuations
* **peak prominence** → salience of a peak
* **peak width** (often at half-prominence or half-maximum) → duration
* **peak amplitude** (or **peak height**) → magnitude

That's essentially the standard terminology used in signal processing libraries (e.g., SciPy's `find_peaks`) and much of the engineering literature.

> Let's describe the rise of the first prominent speed peak in more detail. The rise starts a the baseline and ends at a local maximum right before the plateau. It show some dips and is therefore not monotonous, right?

Yes, that's right. In mathematics, a **monotonic increase** means the signal never decreases. Even a tiny dip means the rise is **not strictly monotonic**.

However, in signal processing people often distinguish between the **overall trend** and the local fluctuations. So you can describe it more naturally than simply saying "non-monotonic."

Here are some good options, from most technical to most descriptive:

* **Overall increasing trend with small local dips.** ✓
* **Predominantly increasing despite minor fluctuations.** ✓✓
* **A rising edge interrupted by several shallow dips.** ✓✓✓ (very signal-processing)
* **A gradual rise with superimposed fluctuations.** ✓✓✓
* **A noisy rise toward the plateau.** ✓✓ (if the dips are noise rather than meaningful)
* **An undulating rise.** ✓ (more literary, less common in engineering)
* **A non-monotonic rise.** ✓ (mathematically correct, but somewhat dry)

I'd also introduce the term **rising edge**, which is extremely common in electronics and signal processing. Although it often refers to a sharp transition, it's equally understandable for a gradual increase:

> *The first pulse begins with a rising edge from the baseline to a local maximum. Although the overall trend is increasing, the rising edge contains several small dips before reaching a short plateau.*

Or, if the fluctuations are just noise:

> *The rising edge is approximately monotonic, with minor fluctuations around the overall upward trend.*

### Vocabulary for the individual features

Along that rising edge you can identify:

* **baseline**
* **onset** (where the rise begins)
* **rising edge** (the increasing segment)
* **local dips** or **local minima**
* **local maxima** (small bumps)
* **shoulder** (a change in slope before the plateau)
* **plateau onset** (where the plateau begins)

In fact, **shoulder** might describe the end of the rise even better than "local maximum." A **shoulder** is a rounded transition where the curve begins to flatten before reaching or entering a plateau. So if the signal eases into the plateau rather than peaking sharply, I'd write:

> *The rising edge terminates in a shoulder, after which the signal enters a plateau lasting approximately 250 ms.*

That wording is common in spectroscopy, chromatography, and signal analysis.

> Cool! Now, let's say I want to exactly identify the onset index (or time) of the rising edge and the index of the shoulder's local maximum or absolute maximum of the plateau. How would you go about it?

This is a classic **change-point detection** problem. I'd avoid relying solely on the raw signal because your onset is gradual and slightly noisy. Instead, I'd define the landmarks in terms of both the signal and its derivatives.

## 1. Onset of the rising edge

Conceptually:

> **The onset is the first statistically significant departure from the baseline.**

Several practical definitions are common.

### A. Threshold crossing (simple and robust)

Estimate the baseline mean ( \mu ) and standard deviation ( \sigma ).

Define onset as the first sample satisfying

* (x > \mu + k\sigma), with (k \in [2,5]),
* for at least (N) consecutive samples.

This avoids triggering on isolated noise.

---

### B. Positive slope threshold (my favorite)

Compute a smoothed first derivative.

The onset is the first index where

* the slope exceeds a threshold,
* and remains positive for some minimum duration.

This detects the **beginning of the rising edge**, not merely when the signal has already become large.

---

### C. Piecewise-linear breakpoint

Fit

* a horizontal baseline,
* followed by a rising line.

The breakpoint is the onset.

This is probably the cleanest definition if you need repeatability.

---

## 2. Shoulder

The shoulder is trickier because it isn't a sharp peak.

I'd define it as

> **the transition from the rising edge into the plateau.**

Several possibilities exist.

### A. First local maximum before the plateau

Exactly what you suggested.

Advantages:

* easy
* intuitive

Disadvantage:

* sensitive to small fluctuations.

---

### B. First derivative approaches zero

The shoulder occurs when

[
\frac{dx}{dt}
]

drops below some small threshold after the rise.

This matches the geometric meaning of a shoulder.

---

### C. Maximum curvature

Compute the second derivative.

The shoulder often coincides with the largest decrease in slope before the plateau.

Very elegant mathematically, but second derivatives amplify noise.

---

## 3. Plateau maximum

If the plateau isn't perfectly flat, I'd simply define

> **the plateau maximum as the absolute maximum within the plateau interval.**

If there are several identical maxima, choose

* the first,
* the center,
* or the last,

and state the convention.

---

## What I would use for your signal

Looking at your orange trace, I'd probably implement something like this:

1. Smooth with a Savitzky–Golay filter (preserves peak shape well).
2. Estimate the baseline from the pre-onset region.
3. Detect onset using a **slope threshold** plus a **minimum duration**.
4. Detect the shoulder as the **first local maximum** after onset whose derivative remains close to zero for some time afterward.
5. Define the plateau maximum as the **highest sample within the detected plateau**.

This approach is robust to the little dips on the rising edge.

---

I also like to distinguish between the **geometric landmark** and the **algorithm**:

| Landmark            | Geometric definition            | Practical detector                                                         |
| ------------------- | ------------------------------- | -------------------------------------------------------------------------- |
| **Onset**           | First departure from baseline   | Baseline + slope threshold                                                 |
| **Rising edge**     | Increasing segment              | Between onset and shoulder                                                 |
| **Shoulder**        | Transition from rise to plateau | First sustained low-slope region or first local maximum before the plateau |
| **Plateau**         | Approximately constant region   | Low derivative for a minimum duration                                      |
| **Plateau maximum** | Highest point on the plateau    | Absolute maximum within the plateau                                        |

I find this separation useful because it lets you describe the signal independently of the implementation. In a paper, you can first define the landmarks geometrically ("the shoulder marks the transition from the rising edge to the plateau") and then state how they were detected algorithmically ("detected as the first local maximum followed by at least 100 ms of low slope"). This makes your methodology both intuitive and reproducible.

> Let's concentrate on onset first. How would I implement option A. Threshold crossing  in Python/NumPy with `speed` being a 1D ndarray?
