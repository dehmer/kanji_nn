Topic: Pen-Up
Bot: ChatGPT
Date: 26 07 01

> Say I want to train a LSTM model with stroke date for Japanese Kanji characters. Besides x and y coordinate, delta x and delta y, as well as euclidean distance and heading for each point, a pen-down (1) pen up (0) data column determines the end of a stroke. Now I see three options on how pen-up can be used.

> A. pen-up is on the last point of a stroke, which is different from the second to last.
> B. the last point of a stroke is duplicated with first instance down-down (1) and the copy with pen-up (0).
> C. the pen-up row is not an actual point, but a zero row (on all columns).

> Which is an appropriate approach? Are there alternatives not mentioned?

This is a sequence modeling question, and there isn't a universally "correct" representation. The best choice depends on what you want the LSTM to learn and whether the pen state is an input, an output, or both.

Here's how the three options compare.

| Option | Description                                                       | Advantages                                           | Disadvantages                                                                          |
| ------ | ----------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **A**  | Last real point has pen-up=0                                      | No duplicated or fake points                         | The pen state changes at a spatial point, which can be slightly ambiguous.             |
| **B**  | Duplicate last point; first copy pen-down=1, second copy pen-up=0 | Explicit "lift pen" event; common in sequence models | Adds duplicate points and slightly lengthens sequences.                                |
| **C**  | Zero vector marks pen-up                                          | Very explicit sequence boundary                      | Artificial point far from data distribution unless masked. Usually the weakest option. |

## My recommendation

I would choose **B**.

The pen lifting is an **event**, not a location. By duplicating the last coordinate, you separate

```
(x,y)  pen touching paper
```

from

```
same (x,y), pen lifted
```

The network learns

> "After arriving here, lift the pen."

This is similar to how many handwriting-generation datasets encode end-of-stroke.

For example

```
x    y    dx   dy   dist heading pen

10   20   ...  ...  ...  ...     1
13   21   ...  ...  ...  ...     1
15   25   ...  ...  ...  ...     1
15   25   0    0    0    0       0   <-- lift pen
```

Notice that the duplicated point naturally has

* dx = 0
* dy = 0
* distance = 0
* heading = unchanged or 0 (pick one convention consistently)

This makes physical sense.

---

## Why I don't like C

A zero row

```
0 0 0 0 0 0 0
```

is not a pen lift.

It's a point at the origin.

Unless you explicitly mask it out, the LSTM must learn that

```
(0,0)
```

sometimes means

> "ignore this; it's actually end-of-stroke."

That is an unnecessary burden.

Even if your coordinates are normalized, the artificial vector does not resemble the rest of the data.

---

## Option A is acceptable

Many online handwriting datasets simply label the last point

```
...
(15,25) pen=1
(18,28) pen=0
```

meaning

> "this is the last sampled point before lifting."

This works perfectly well.

The only drawback is that the same sample represents both

* a location
* an event

whereas option B separates those concepts.

---

# An alternative I'd actually consider

Instead of treating pen-up as another point, make it its own token.

Your vocabulary becomes

```
POINT
POINT
POINT
<EOS_STROKE>
POINT
POINT
<EOS_STROKE>
...
```

The stroke-end token has no coordinates at all.

Conceptually this is cleaner.

For an LSTM, this can be encoded as

```
[x, y, dx, dy, dist, heading, pen]
```

where

```
pen = 1   -> regular point
pen = 0   -> stroke-end token
```

and the coordinate features are masked or ignored rather than set to arbitrary values. This avoids introducing fake spatial data.

---

# Another alternative: predict pen state separately

Many handwriting models treat pen state as an independent output.

Each time step predicts

```
(x_next, y_next)
```

and simultaneously

```
P(pen_up)
```

or

```
P(end_of_stroke)
```

During training, every row is a real point. The target for each point includes whether the pen lifts afterward. This avoids duplicate points while giving the model a dedicated learning signal for stroke termination.

---

# One more consideration

Since you're already computing

* Δx
* Δy
* distance
* heading

you're encoding movement twice:

* `(x, y)` gives absolute position.
* `(Δx, Δy)` gives relative movement.
* `distance` and `heading` are deterministic functions of `(Δx, Δy)`.

This redundancy isn't necessarily harmful, but it may not provide additional information. Many successful handwriting models use only one of these representations:

* absolute coordinates `(x, y)`,
* relative displacements `(Δx, Δy)` (very common because they improve translation invariance), or
* `(distance, heading)`.

Including all four movement-related features increases input dimensionality and can encourage the network to learn duplicate relationships rather than the underlying stroke dynamics.

## Overall recommendation

For an LSTM trained on kanji stroke sequences:

1. Use **real sampled points only** whenever possible.
2. Represent stroke termination either as:

   * **Option B** (duplicate last point with a pen-up event), or
   * a dedicated end-of-stroke token (cleaner if your architecture supports masking or special tokens).
3. Avoid a zero-vector row unless your preprocessing explicitly masks it so the model never interprets it as a genuine spatial observation.

If your goal is handwriting generation rather than classification, the dedicated stroke-end event (either as a duplicated point with a pen-up flag or as its own token) generally provides the clearest training signal.

> Ok. I consider Option B. Concerning Δx and Δy: I understand Δx and Δy (both starting with 0) are "enough" and x, y, distance and heading should be dropped from tensor matrix. Is this correct or the preferred way?

Yes, that would generally be the preferred approach, especially for handwriting modeling.

The reason is that **Δx and Δy contain the essential information about the pen trajectory**, while the other features are either redundant or can make learning harder.

Here's how the features relate:

* **x, y**: absolute position
* **Δx, Δy**: relative movement
* **distance** = √(Δx² + Δy²)
* **heading** = atan2(Δy, Δx)

So if the model receives `(Δx, Δy)`, it can, in principle, infer distance and direction from them. Conversely, `(distance, heading)` can reconstruct `(Δx, Δy)`.

### Why Δx and Δy are usually preferred

Using relative displacements has several advantages:

* **Translation invariance.** A kanji drawn in the upper-left and the same kanji drawn in the center produce nearly identical sequences.
* **Smaller numerical range.** Δ values are usually much smaller than absolute coordinates, which often makes optimization easier.
* **Focus on stroke dynamics.** The network learns how the pen moves rather than where it happens to be on the page.

This is why many classic online handwriting models (including handwriting generation work) use only relative displacements together with the pen state.

### What would your input look like?

For Option B, a sequence could be

| Δx |  Δy | pen |                                                           |
| -: | --: | :-: | --------------------------------------------------------- |
|  0 |   0 |  1  |                                                           |
|  3 |   1 |  1  |                                                           |
|  2 |   4 |  1  |                                                           |
|  0 |   0 |  0  | ← duplicated last point, pen lifted                       |
| -8 | -15 |  1  | ← first point of next stroke (relative to previous point) |

Notice that the jump to the first point of the next stroke is simply the displacement between the last point of the previous stroke and the first point of the next stroke. The pen state tells the network that this movement occurred with the pen lifted.

### One refinement

I would **not** force the first point of every stroke to have `(Δx, Δy) = (0, 0)`.

Instead:

* The **very first point of the entire character** naturally has `(0,0)` because there is no previous point.
* The **first point of subsequent strokes** should have the actual displacement from the previous stroke's last point.

For example:

```
Stroke 1:
(10,10)
(13,11)
(15,14)

Stroke 2:
(8,5)
(9,7)
```

becomes

| Δx | Δy | pen |                                       |
| -: | -: | :-: | ------------------------------------- |
|  0 |  0 |  1  |                                       |
|  3 |  1 |  1  |                                       |
|  2 |  3 |  1  |                                       |
|  0 |  0 |  0  | ← pen-up event                        |
| -7 | -9 |  1  | ← move through the air to next stroke |
|  1 |  2 |  1  |                                       |

This is actually informative: the model learns not only the shape of each stroke but also where the next stroke begins relative to the previous one.

### My recommendation

For kanji, I would use just **three input features**:

* Δx (normalized)
* Δy (normalized)
* pen state (1 = pen touching paper, 0 = pen lifted)

This representation is compact, avoids redundant information, and preserves both the stroke geometry and the stroke order. It is also very similar to the representations used in well-known online handwriting sequence models.

> Thanks! Very informative and helpful, indeed.

You're very welcome! This is an interesting problem, and I think you're on a good track.

One additional suggestion that often pays off with kanji is **normalization**. Since different writers produce characters at different sizes and positions, it's usually beneficial to normalize each character before computing Δ values. A common pipeline is:

1. Translate the character so its bounding box or centroid is centered at the origin.
2. Scale it so the larger of width or height becomes 1 (or another fixed size).
3. Compute Δx and Δy from the normalized coordinates.

This makes the model focus on stroke order and shape rather than absolute size.

Another subtle point concerns the **sampling density**. If one writer contributes 50 points for a stroke and another contributes 200, the LSTM will see quite different sequence lengths for the same shape. Resampling each stroke to a more uniform point spacing (or a fixed spatial interval) often improves consistency and training efficiency.

Finally, if you later move beyond a plain LSTM to a bidirectional LSTM or a Transformer, the same representation—`(Δx, Δy, pen)`—remains a solid choice, so your data format won't need to change.

It sounds like you're building a dataset yourself. If so, I'd also recommend keeping the original `(x, y)` coordinates in your source data, even if the network never sees them. They're invaluable for:

* visualizing samples during debugging,
* verifying preprocessing,
* experimenting with different feature representations later,
* and reconstructing the strokes for qualitative evaluation.

Good preprocessing is often the difference between an LSTM that struggles and one that converges cleanly. In sequence models for handwriting, the data representation is every bit as important as the network architecture itself.

I wish you success with your kanji project—it sounds like a fascinating application of sequence modeling.

> Thank! Normalization and resampling are already part of the pipeline for test and verification data.

Excellent—that's reassuring to hear. Those two steps eliminate two of the biggest sources of variation in online handwriting data.

Given what you've described, your pipeline is already close to what I would consider a solid foundation:

1. Normalize the character (translation + scaling).
2. Resample strokes to approximately uniform point spacing.
3. Compute Δx and Δy.
4. Encode stroke ends using your Option B (duplicated last point with pen-up).
5. Feed `(Δx, Δy, pen)` into the LSTM.

One thing I'd keep an eye on is the **distribution of Δ values** after resampling. Ideally, most Δx and Δy values should be relatively small and centered around zero. If you find a few large jumps (typically between strokes), that's expected—they represent pen movement through the air.

Another idea, if you eventually experiment with generation rather than recognition, is to replace the binary pen feature with a **3-state pen encoding**:

* **0** = pen touching paper (continue stroke)
* **1** = pen lifted (between strokes)
* **2** = end of character (EOS)

That gives the model an explicit "I'm done" signal instead of relying on sequence length or padding. For fixed-length classification tasks, this isn't necessary, but for sequence generation it's often useful.

One final curiosity: are you training the LSTM to **recognize** kanji (sequence → character label), or to **generate/predict** stroke sequences (or perhaps to complete partially drawn characters)? The optimal loss function and even the best sequence encoding differ slightly depending on that goal.

> OK. Currently I am looking at classification topic. As a side note: Input pipeline also includes Gaussian smoothing which should also keep Δx, Δy in check.

That makes perfect sense. For **classification**, your representation is actually simpler than what's needed for handwriting generation, since the model only needs to extract discriminative features from the sequence rather than reproduce it.

Gaussian smoothing is also a sensible addition. It has several benefits:

* It reduces digitizer noise and hand tremor.
* It makes the local direction changes more consistent.
* It reduces spurious high-frequency oscillations in Δx and Δy.
* It often improves the consistency of the resampling step.

One thing to watch is the **order of preprocessing**. A typical sequence is:

1. Raw points
2. Gaussian smoothing
3. Resampling
4. Normalization (or normalize before smoothing if scaling varies greatly)
5. Compute Δx, Δy
6. Encode pen state

Some people smooth after resampling, but smoothing first generally preserves the intended trajectory before you interpolate.

### For classification, one more feature is worth considering

Although I'd start with only `(Δx, Δy, pen)`, there is one additional feature that is **not redundant**:

**Normalized arc length (or time index)**

This tells the network where along the character it currently is. For example:

```
Δx   Δy   pen   s
```

where

* `s = 0.0` at the first point,
* `s = 1.0` at the last point.

Unlike `distance` or `heading`, this cannot be reconstructed from Δx and Δy. Sometimes it helps the model distinguish shapes that are locally similar but occur at different stages of writing.

I wouldn't add it initially, but it's a worthwhile ablation experiment later.

### I'd also recommend one experiment

Since you're clearly building a careful pipeline, I'd train two otherwise identical models:

* **Model A:** `(Δx, Δy, pen)`
* **Model B:** `(Δx, Δy, pen, stroke_index)`

where `stroke_index` is normalized to `[0,1]` (or embedded as an integer if the maximum number of strokes is known).

Kanji are highly structured, and the stroke order itself is informative. Giving the network an explicit notion of "this is stroke 1 of 8" versus "stroke 7 of 8" can sometimes improve classification without much extra complexity.

### Overall impression

From everything you've described—Gaussian smoothing, resampling, normalization, relative coordinates, and explicit pen-up events—your preprocessing is much closer to what you'd see in a research prototype than in a typical first implementation. At this point, I'd resist the temptation to add many more handcrafted features. Let the LSTM discover higher-level patterns from a clean, compact representation.

In machine learning, especially with sequence models, a small set of well-chosen, physically meaningful features often outperforms a larger set of redundant ones. I think your current design is a good example of that philosophy.

> Nice! Since I also have timestamps for all points from input, I can derive normalized point indices.

Yes—and that's actually an interesting opportunity because **point index** and **timestamp** capture different aspects of the writing process.

If you have timestamps, you can derive several useful quantities:

* **Normalized time** (`t ∈ [0,1]`) — where you are in the writing of the character.
* **Δt** — time elapsed between consecutive sampled points.
* **Velocity** (`√(Δx²+Δy²)/Δt`).
* **Acceleration** (change in velocity).

For classification, I'd rank their usefulness roughly as follows:

1. **Δx, Δy, pen** (your baseline)
2. **Normalized time or point index** (worth experimenting with)
3. **Δt** (if your sampling rate is not constant)
4. **Velocity/acceleration** (only if timing is reliable)

### Point index vs. timestamp

If your tablet samples at a **fixed frequency** (e.g., every 5 ms), then normalized point index and normalized time are nearly identical. In that case, I'd just use the normalized index because it's simpler.

If the device reports **actual timestamps** and the sampling rate varies—as is common on many tablets—then normalized time carries genuine information. Writers naturally slow down on hooks, corners, and stroke endings, and those pauses can help distinguish similar characters.

### One caution

Since you're already **resampling spatially**, be careful about using the resampled points' indices as a proxy for time.

Imagine a stroke:

```
Raw:
fast -------- slow curve ---- fast
```

After resampling to equal spatial intervals, you'll have many points along the long sections regardless of how quickly they were written. The original timing information is no longer represented by the point spacing.

If you want to preserve temporal dynamics, a good approach is:

* Smooth the trajectory.
* Resample the geometry.
* **Interpolate the timestamps onto the new points.**

That way each resampled point still has an estimated writing time, and you retain speed information while benefiting from uniform spatial sampling.

---

This actually raises an interesting research question for your project. There are two fundamentally different representations:

* **Geometry-driven:** uniform spatial resampling, emphasizing shape.
* **Time-driven:** original sampling, emphasizing writing dynamics.

For **offline recognition**, geometry usually wins because only the final shape is available.

For **online kanji recognition**, where you have stroke order and timing, the dynamics can add discriminative power. Some pairs of characters have very similar geometry but are written with noticeably different motion patterns.

If you have enough data, it would be a worthwhile experiment to compare:

* **Model G:** geometry only (`Δx, Δy, pen`)
* **Model GT:** geometry + normalized time (or Δt)

My guess is that geometry will account for most of the accuracy, but timing may provide a modest improvement, particularly for cursive or naturally written kanji where the motion pattern carries additional information.

> OK, I will check how not not mess up timestamps (or Δt) while smoothing/resampling.

That's a good instinct. In fact, I'd recommend treating the **geometry** and the **timing** as two related but distinct signals.

A few practical approaches, in increasing order of sophistication:

### 1. Interpolate timestamps onto the resampled points (recommended)

Suppose your original points are

|   x |   y |     t |
| --: | --: | ----: |
| 0.0 | 0.0 |  0 ms |
| 0.4 | 0.2 |  8 ms |
| 1.1 | 0.7 | 17 ms |
| 2.0 | 1.3 | 30 ms |

After smoothing and resampling, each new point lies somewhere along the original polyline. You can compute its position along the curve (arc length) and interpolate **t** between the neighboring original samples.

This preserves the original writing rhythm reasonably well.

---

### 2. Smooth only the geometry

Another common approach is:

* Smooth **x, y**.
* Leave timestamps untouched.
* Interpolate timestamps only when creating new points.

This avoids "smoothing time," which usually has no physical meaning.

---

### 3. Never smooth timestamps

I'd avoid applying a Gaussian filter directly to timestamps. Time is already monotonic and typically measured accurately enough. Smoothing it can produce artificial velocity profiles.

---

## One subtle issue

Spatial resampling changes the interpretation of Δt.

Suppose the writer pauses before making a hook:

```
Original:

----->----->-----> (pause) ---->
```

The timestamps reflect the pause.

After uniform spatial resampling, the points are equally spaced in distance, but one interval may have a much larger Δt.

That's actually *good*. It tells the model:

> "The pen moved the same distance here, but much more slowly."

So Δt becomes a measure of **local speed**, which can be informative.

---

## Something I'd compute once

Since you're already building a preprocessing pipeline, I'd compute cumulative arc length:

[
s_i = \sum_{k=1}^{i} \sqrt{\Delta x_k^2 + \Delta y_k^2}
]

Then normalize it:

[
\hat{s}*i = \frac{s_i}{s*{\text{final}}}
]

Now every point has:

* normalized position along the trajectory (`ŝ`)
* normalized time (`t̂`)

Comparing these two is surprisingly informative:

* if `t̂ ≈ ŝ`, the stroke was written at nearly constant speed;
* if `t̂ > ŝ`, the writer slowed down early;
* if `t̂ < ŝ`, the writer accelerated.

That relationship encodes writing style without introducing many extra features.

---

I also have one suggestion for your experimental methodology. Since you're clearly engineering the input carefully, resist changing several things at once. Start with the simplest baseline:

* `(Δx, Δy, pen)`

Then add **exactly one** new feature at a time:

1. * normalized time
2. * Δt
3. * stroke index
4. * anything else

Keep the architecture fixed while doing this. You'll quickly discover which features genuinely improve classification and which merely add complexity. In my experience, those ablation studies are often more valuable than trying to design the "perfect" feature set from the outset. They also make it much easier to explain and justify your design choices if you later write up the project or compare models.

> Right! For me this is more of an explorational field. I have next to no knowledge about RNNs or deep learning in general. On the other hand, that's pure fun and a chance to learn a few things.

That's actually a great position to be in. You're approaching it with curiosity rather than trying to reproduce someone else's architecture blindly.

From our discussion, I also get the impression that your strength is in **thinking carefully about the data representation**. That's a valuable skill because, in sequence learning, the input representation often has a larger impact than changing the LSTM from, say, 2 to 3 layers.

One piece of advice I'd give someone starting with RNNs is:

> **Don't optimize the network until you're convinced the data pipeline is sound.**

In practice, I'd spend far more time verifying the preprocessing than tuning hyperparameters.

For example, I would make it easy to visualize every preprocessing stage:

```
Raw points
    ↓
Smoothed points
    ↓
Resampled points
    ↓
Δx, Δy vectors
    ↓
Final tensor
```

Being able to plot these stages for a handful of characters is incredibly useful. You'll catch issues that no loss curve will reveal—an off-by-one error in Δ computation, a duplicated point at the wrong place, a stroke boundary encoded incorrectly, and so on.

Another thing that's easy to overlook: **overfit on purpose.**

Take a tiny dataset—perhaps just 50 or 100 kanji samples—and train until the model reaches nearly 100% training accuracy. If it can't do that, the problem is usually not that the network is "too weak"; it's often a bug in the preprocessing, labels, sequence handling, or training loop. This simple sanity check has saved many researchers a lot of time.

Since you're learning, I'd also recommend building your intuition by inspecting what the network "sees." If you print a few input sequences, you should be able to read them yourself:

```
Δx   Δy   pen
 0    0    1
 2    1    1
 1    3    1
 0    0    0
-8  -12    1
...
```

After a while, you'll start recognizing patterns like "that's the start of the second stroke" or "that looks like a left-falling stroke." Developing that intuition makes it much easier to reason about what the LSTM might learn.

Finally, don't be surprised if your first reasonably configured LSTM performs better than you expect. Online handwriting recognition has several characteristics that LSTMs are naturally good at:

* The data is sequential.
* The order of points matters.
* Stroke order carries information.
* Dependencies span many time steps.

That's exactly the kind of problem LSTMs were designed for.

One prediction, based on the questions you've been asking: I suspect that six months from now you'll be spending more time experimenting with preprocessing and feature engineering than with the LSTM itself. That's not a bad thing—in handwriting recognition, it's often where the biggest gains come from. The network architecture may change only slightly, while the data pipeline evolves considerably as you learn what helps the model generalize. That's a very natural progression in projects like this.

> I probably want to look at a few papers which deal with character classification based on vector data. There are tons on raster graphics, but I have not found much on vectors. Do you have a few suggestions? Preferable fairly understandable for an average Joe.

I think there is a small "trap" here. If you search for **"handwriting recognition"**, about 95% of the papers you'll find are about OCR (images). What you're doing is **online handwriting recognition** (sometimes also called *trajectory-based* or *stroke-based* recognition), which is a much smaller but very interesting field.

Given what you've told me, I'd recommend the following reading order.

### 1. Alex Graves — *Unconstrained On-line Handwriting Recognition with Recurrent Neural Networks* (2007)

This is the paper I would start with. It is one of the foundational papers for applying LSTMs to online handwriting. It uses pen trajectories rather than images and explains why recurrent networks are a natural fit for this type of data. Don't worry if the CTC parts are unfamiliar—you can safely skim those on a first reading. ([ResearchGate][1])

### 2. Alex Graves — *Generating Sequences With Recurrent Neural Networks* (2013)

Although this paper is about handwriting **generation**, not classification, I think you'll enjoy it because you'll immediately recognize the data representation. The network is fed sequences of `(Δx, Δy, pen-state)`—essentially the same representation you've converged on. It's one of those papers where you can say, "Ah, so that's why this encoding is popular." ([arXiv][2])

### 3. Victor Cărbune et al. — *Fast Multi-language LSTM-based Online Handwriting Recognition* (2020)

This is much more modern and describes a production-quality system. It introduces some engineering ideas, such as Bézier curve representations for speed, but it also discusses practical feature representations and network choices. I wouldn't read this first, but it's an excellent "where did the field end up?" paper. ([Springer][3])

### 4. Felix Ott et al. — *Benchmarking Online Sequence-to-Sequence and Character-based Handwriting Recognition* (2022)

This paper is valuable because it compares different architectures—LSTMs, temporal convolutional networks, and Transformers—on online handwriting data. Even if you stay with LSTMs, it's useful to see how they compare to newer models. ([Springer][4])

---

## A paper I think you'll especially appreciate

Knowing how you're approaching this, I'd actually suggest reading Graves' generation paper *before* many classification papers.

Why?

Because the paper spends time discussing the representation of handwriting as a sequence:

* pen-up / pen-down
* relative coordinates
* sequence modeling
* variable-length inputs

Those are exactly the design questions you've been asking over the last few days.

---

## Also look for these datasets

Even if you don't use them, the accompanying papers are informative.

* **IAM-OnDB** — probably the most famous online handwriting dataset. Many papers benchmark on it. ([Springer][3])
* **CASIA Online Handwriting Database** — contains Chinese online handwriting and is very relevant if you're interested in kanji, because the stroke structures are closely related. ([Springer][3])

---

## A keyword that will save you hours

When searching, use combinations like:

* `"online handwriting recognition LSTM"`
* `"trajectory-based handwriting recognition"`
* `"stroke-based character recognition"`
* `"pen trajectory neural network"`
* `"online Chinese character recognition"`
* `"online Kanji recognition"`

Avoid just searching for "handwriting recognition" unless you specifically want OCR papers.

---

## Finally, one book

This isn't specifically about handwriting, but I think you'd enjoy it:

[Deep Learning with Python (François Chollet)](https://www.manning.com/books/deep-learning-with-python-second-edition?utm_source=chatgpt.com)

It's one of the more approachable introductions to deep learning. It doesn't assume you're a mathematician, and it develops an intuition for why neural networks work rather than drowning you in notation.

---

### One paper I suspect you'll like even more

Based on our conversation, I actually think there is another category of papers you'll enjoy: **online Chinese character recognition**. Those papers deal with thousands of classes, strict stroke order, pen trajectories, and sequence models—the problem is much closer to your kanji classifier than English cursive handwriting is.

If you're interested, I can dig out **5–10 particularly good papers on online Chinese/Japanese character recognition** (roughly 2015–2024), with a short summary of each and an indication of which ones are beginner-friendly versus more mathematical. I think that would be a very targeted reading list for your project.

[1]: https://www.researchgate.net/publication/221619960_Unconstrained_On-line_Handwriting_Recognition_with_Recurrent_Neural_Networks?utm_source=chatgpt.com "(PDF) Unconstrained On-line Handwriting Recognition with Recurrent Neural Networks."
[2]: https://arxiv.org/abs/1308.0850?utm_source=chatgpt.com "Generating Sequences With Recurrent Neural Networks"
[3]: https://link.springer.com/article/10.1007/s10032-020-00350-4?utm_source=chatgpt.com "Fast multi-language LSTM-based online handwriting recognition | International Journal on Document Analysis and Recognition (IJDAR) | Springer Nature Link"
[4]: https://link.springer.com/article/10.1007/s10032-022-00415-6?utm_source=chatgpt.com "Benchmarking online sequence-to-sequence and character-based handwriting recognition from IMU-enhanced pens | International Journal on Document Analysis and Recognition (IJDAR) | Springer Nature Link"

> You are correct! Six months is a realistic estimate ;-)

😂 I had a hunch.

The reason is not that you asked "beginner questions." Quite the opposite: your questions have gradually shifted from

> "How do I use an LSTM?"

to

> "What information should my representation preserve?"

That's a very healthy progression.

If I had to guess, here's what your next six months might look like:

* Month 1: "I'll just train an LSTM."
* Month 2: "Why does changing the resampling interval from 1.0 mm to 0.5 mm improve accuracy by 3%?"
* Month 3: "I wonder if Bézier fitting would preserve curvature better..."
* Month 4: "Should I normalize each stroke individually or the whole character?"
* Month 5: "Maybe bidirectional LSTMs..."
* Month 6: "I need to rerun all 48 experiments because I changed one preprocessing parameter."

Welcome to machine learning research. 😄

---

On a more serious note, I think you have one advantage that many newcomers don't: you're **not treating the neural network as magic**.

You're asking questions like:

* What exactly is the meaning of a pen-up event?
* Which features are redundant?
* Should the network see absolute or relative coordinates?
* How does resampling affect timing?

Those are the kinds of questions that lead to a robust system. The LSTM is ultimately just a function approximator—it can only learn from the information you present to it.

---

One thing I wanted to mention after thinking about your kanji project: **don't underestimate classical baselines.**

Before training an LSTM, it can be surprisingly enlightening to build something very simple:

* flatten the `(Δx, Δy, pen)` sequence after padding,
* train a small MLP or even a nearest-neighbor classifier,
* compare it with the LSTM.

If the LSTM doesn't outperform the simple model, that's a strong signal to inspect the preprocessing or the training setup. Conversely, if it does, you'll have a much better appreciation for *why* sequence models help.

---

Regarding papers, I found one more that I think fits your interests particularly well. The survey **"Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark"** is centered on Chinese characters, but many of the challenges—stroke order, trajectory information, and large character sets—carry over directly to kanji. Even if some sections focus on image-based methods, the discussion of online recognition and datasets provides useful context. ([ScienceDirect][1])

---

And finally, a suggestion that has nothing to do with neural networks.

Keep a little **lab notebook** (Markdown, a text file, Jupyter notebook—anything). Every experiment gets one entry:

* preprocessing settings,
* input features,
* network architecture,
* optimizer and learning rate,
* training/validation accuracy,
* observations ("overfits after 15 epochs", "confuses 木 and 本", etc.).

It sounds mundane, but after 30 or 40 experiments it's worth its weight in gold. You stop relying on memory and start building an intuition based on evidence.

I have to say, I've enjoyed this discussion. It's refreshing to talk with someone who is approaching the subject from the perspective of signal representation and geometry rather than immediately asking, "Which transformer should I use?" I suspect your background is in software engineering or mathematics rather than machine learning, and that's often an excellent foundation for this kind of project.

[1]: https://www.sciencedirect.com/science/article/pii/S0031320316302187?utm_source=chatgpt.com "Online and offline handwritten Chinese character recognition: A comprehensive study and new benchmark - ScienceDirect"

> Top tip! Thanks! Actually, I am writing a small Flutter app to practice Kanji writing (accuracy AND speed) on iPad based on KanjiVG. Comparing single strokes with DTW is easy enough. But I thought to add some more "freehand" experience. But that's a topic for another time.

That’s a very natural extension of what you’re already doing, and honestly a nice use case for this kind of data.

If you already have **KanjiVG stroke templates + DTW per stroke**, you’ve covered the “structural correctness” side pretty well. The interesting part you’re now hinting at is the *continuous experience layer*—making it feel less like a checker and more like handwriting feedback.

A few thoughts that might be useful when you get back to it:

If you move toward a more “freehand” evaluation, you’ll likely end up blending three signals:

* **Geometry match** (what you already do with DTW per stroke)
* **Stroke order / segmentation correctness**
* **Temporal behavior** (speed, hesitation, smoothness)

That last one is where things get interesting but also tricky.

For example, even simple metrics like:

* velocity profile smoothness
* pause duration at stroke starts/ends
* variance in Δx/Δy magnitude

can already give a surprisingly good “feel score” without needing a heavy model.

Another idea that pairs nicely with your current setup: instead of only aligning each stroke independently with DTW, you can occasionally try **pen-lift–aware global alignment** (treat the full character trajectory as one sequence but allow penalties for stroke transitions). That starts to bridge the gap between “strict correction mode” and “natural writing flow”.

And since you're already in Flutter on iPad, you’re in a great position to eventually do something more ambitious like:

* real-time feedback overlay (heatmap of deviation from ideal stroke path)
* replay comparison (user stroke vs canonical stroke animation)
* speed vs accuracy tradeoff scoring (very motivating in practice apps)

But you're right—this is a rabbit hole of its own.

For now, your instinct to separate:

* **clean stroke-level comparison (DTW)**
* from **freehand experience scoring**

is exactly the right architectural boundary. It keeps your system understandable instead of turning it into a single opaque “magic score.”

Whenever you come back to it, I’d be happy to help sketch a clean scoring model that combines these signals without making the system overly complex.
