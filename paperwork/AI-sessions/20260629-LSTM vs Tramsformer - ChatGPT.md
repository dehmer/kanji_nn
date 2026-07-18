Topic: LSTM vs Transformer
Bot: ChatGPT
Date: 26 06 29

> Would you use a LSTM RNN or a a fully connected Transformer to classify roughly 6000 kanji? The input is comprised of 2D vector data, n point for m strokes per kanji. Maximum sequence length of input data (total number of points per kanji) will be limited to 80 but can be less for simpler kanji with less strokes. An additional input column feature will be 1 for pen-down and 0 pen-up for the last point of each stroke.
Given your setup, I would lean toward a **Transformer**, but not a large language model–style Transformer. A relatively small encoder-only Transformer is likely to outperform an LSTM for this task, assuming you have enough training data.

Here's how I would think about it.

### Your data is naturally sequential

Each sample is essentially

```
(x, y, pen_state)
```

repeated for up to 80 points.

So your input shape is

```
(max_len=80, features=3)
```

where

* x
* y
* pen_up/down flag

The sequence length is actually very short by Transformer standards.

---

## LSTM advantages

LSTMs work well because:

* They naturally process drawing order.
* They have relatively few parameters.
* They can perform well on modest datasets.
* They don't require positional encodings since order is implicit.

For sequences of only 80 steps, vanishing gradients are generally not a serious issue.

If you only had a few thousand examples, I'd probably recommend an LSTM.

A good LSTM architecture might be

```
Input
↓

2-layer BiLSTM
256 hidden units

↓

Attention pooling

↓

Dense

↓

6000-way softmax
```

Adding attention pooling usually helps significantly.

---

## Why I still prefer a Transformer

Your problem has characteristics that Transformers handle particularly well.

### 1. Long-range geometric relationships

Kanji often contain components that are spatially separated.

For example,

```
left radical
...
many points later
...
right radical
```

An LSTM has to compress everything through its hidden state.

A Transformer can directly compare any point with any other point.

---

### 2. Sequence is short

80 tokens is tiny.

Self-attention cost is

```
80² = 6400
```

which is essentially negligible on modern hardware.

So one of the biggest drawbacks of Transformers disappears.

---

### 3. Stroke interactions

Suppose two strokes are parallel.

The Transformer can immediately learn

```
stroke 3 ↔ stroke 8
```

whereas an LSTM has to propagate information through every intermediate timestep.

---

### 4. Classification size

6000 classes is large.

Transformers tend to produce richer embeddings before the classifier.

That generally helps with large-class problems.

---

## Fully connected Transformer?

If by "fully connected" you mean standard self-attention over all points:

Yes.

Since the sequence length is only 80, there is no need for sparse attention or local attention.

---

## Input representation

Instead of just

```
x
y
pen
```

I'd strongly consider

```
Δx
Δy
pen
```

or

```
x
y
Δx
Δy
pen
```

Relative motion often generalizes better because it reduces sensitivity to absolute position. You can also normalize each character to a consistent bounding box while preserving aspect ratio before computing deltas.

---

## Positional encoding

You still need positional information because point order matters.

I'd use learned positional embeddings rather than sinusoidal ones.

---

## Model size

Because sequences are so short, the model can remain quite small.

For example:

```
Input (80 × 5)

↓

Linear projection

↓

6 Transformer encoder layers

hidden size 256

8 attention heads

↓

CLS token

↓

MLP

↓

6000-way softmax
```

This would likely have only a few million parameters.

---

## Even better: treat strokes separately

One idea I think is particularly well suited to online handwriting is a hierarchical approach:

```
points
↓

stroke encoder

↓

stroke embeddings

↓

Transformer over strokes

↓

classifier
```

This reflects the natural structure of kanji. The point-level encoder captures the geometry of each stroke, while the higher-level Transformer models relationships among strokes. It also avoids mixing unrelated points across pen lifts in the earliest layers.

---

## What I would build

For a new system, I'd probably choose:

* Normalize each kanji to a standard coordinate system.
* Use features `(x, y, Δx, Δy, pen_flag)`.
* Add a learned positional embedding.
* Use a small Transformer encoder (4–8 layers, hidden size around 256–384).
* Use a `[CLS]` token or attention pooling for aggregation.
* Train with cross-entropy over the ~6000 classes.

If your dataset is relatively small (for example, fewer than about 100,000 labeled examples), a **BiLSTM with attention** is also a strong baseline and may be easier to optimize. Once you have a moderately large dataset (hundreds of thousands to millions of examples), I would generally expect the Transformer to achieve higher accuracy because it can better model interactions between distant parts of the drawing while incurring very little computational overhead at an 80-point sequence length.

> Would it be beneficial to also supply delta-x and delta-y besides x and y coordinates to an LSTM?


Yes—I think it's likely to be beneficial, and it's a common technique in online handwriting recognition.

The intuition is that **absolute position** and **local motion** provide complementary information.

| Feature  | What it tells the model      |
| -------- | ---------------------------- |
| x, y     | Where the pen currently is   |
| Δx, Δy   | How the pen is moving        |
| pen flag | Whether the stroke ends here |

For example, consider these three consecutive points:

```
(10,20)
(11,20)
(12,20)
```

The network has to infer that the pen is moving horizontally from the coordinates alone.

With deltas:

```
x   y   dx  dy
10 20   0   0
11 20   1   0
12 20   1   0
```

the direction is immediately explicit.

## Why this helps an LSTM

An LSTM's hidden state has limited capacity. If it has to first compute

```
Δx = x_t - x_{t-1}
Δy = y_t - y_{t-1}
```

internally, that's representational capacity spent on something deterministic.

Providing those values directly lets the network devote more of its capacity to higher-level structure, such as:

* hooks
* corners
* loops
* stroke endings
* radicals

rather than basic geometry.

This is similar to giving image models edge maps or optical flow in some vision tasks.

## I'd actually use five input features

For each point:

```
x
y
dx
dy
pen_up
```

Optionally normalize all coordinates so every character fits inside something like

```
[-1,1] × [-1,1]
```

or

```
[0,1] × [0,1]
```

before computing the deltas.

## Even better features

If you're willing to engineer a few more features, these are commonly useful:

```
x
y
dx
dy
distance = sqrt(dx²+dy²)
pen_up
```

Distance gives instantaneous speed along the trajectory.

You can also derive

```
heading = atan2(dy, dx)
```

or encode it as

```
sin(theta)
cos(theta)
```

to avoid angle discontinuities. Sharp changes in heading often correspond to corners and stroke transitions.

## What I would try first

I'd start with a very simple feature vector:

```
[x, y, dx, dy, pen_up]
```

and compare it against

```
[x, y, pen_up]
```

Everything else—model architecture, optimizer, training schedule—should stay identical. In handwriting tasks, it's common to see a modest but consistent improvement from including the delta features because they make local stroke dynamics explicit without requiring the network to infer them from successive coordinates.

> Ok, I implied pen-up/down. I should be more thorough in the future. Including distance and heading seems interesting. Is it safe to assume that LSTM and Transformer models would be indifferent about whether Hiragana and Katakana would be trained parallel to Kanji?

That's actually a very good question, and I don't think the answer is "it doesn't matter." I would **train them together**, unless you have a specific reason not to.

Here's why.

### Shared low-level stroke vocabulary

From the network's perspective, it doesn't know what a kanji or hiragana is. It only sees sequences of pen movements.

Many primitives are shared across all Japanese scripts:

* horizontal strokes
* vertical strokes
* left/right diagonals
* hooks
* loops
* stroke endings
* short ticks
* curved motions

For example, the same basic stroke shapes appear in:

* Hiragana
* Katakana
* Kanji

The model benefits from seeing many more examples of these primitives.

---

### Hiragana and katakana act as additional training data

Imagine your classifier has

* ~6,000 kanji
* 46 hiragana
* 46 katakana

Adding the kana only increases the number of classes by about 1.5%.

In return, you get thousands (or millions) of additional examples of handwriting.

Those examples help the network learn better feature detectors.

---

### Similar to ImageNet pretraining

It's analogous to image classification.

Suppose your task is identifying 500 bird species.

Training only on birds works.

Training on birds, dogs, cats, cars, and chairs often produces *better early feature extractors*, because the network first learns generic visual concepts like edges, curves, and textures.

Kana provide the handwriting equivalent of that.

---

### Shared radicals

An even stronger reason is that many kana historically derive from kanji.

For example:

* か resembles parts of kanji-derived cursive forms.
* り resembles stroke patterns found in many kanji.
* Several katakana are derived from components of kanji.

Even if modern forms differ, there is overlap in stroke geometry.

---

### Potential downside

The only real downside is if your dataset becomes heavily imbalanced.

For example:

```
Kana:
5 million samples

Kanji:
100 samples each
```

Then the network may spend much of its capacity optimizing for kana unless you address the imbalance.

This is straightforward to mitigate with techniques such as:

* class-balanced sampling,
* weighted loss functions,
* or ensuring each mini-batch contains a healthy mix of kana and kanji examples.

---

## One thing I would seriously consider

Rather than thinking of this as a **6,092-class problem**, I'd think of it as learning a handwriting embedding.

For example:

```
Input
↓

Encoder (LSTM or Transformer)

↓

256-dimensional embedding

↓

Classifier
```

Once the encoder learns a good embedding, adding more classes (kana, symbols, digits, Latin letters, punctuation) often costs very little. The encoder's job is to map visually similar handwriting close together in feature space, while the classifier maps those embeddings to labels.

---

## An interesting extension

If you're collecting your own data, I'd even consider including:

* Hiragana
* Katakana
* Kanji
* Arabic numerals
* Latin alphabet (upper and lower case)
* Common punctuation

Not because you necessarily want to recognize all of them immediately, but because they enrich the variety of stroke patterns the encoder sees. This can improve the robustness of the learned representation, especially if you later reuse the encoder for other handwriting tasks.

So if I were designing this from scratch, I'd train **one model** over the entire character set and let it learn a unified representation. The additional 92 kana classes are negligible compared with roughly 6,000 kanji classes, while the extra handwriting examples can improve the quality of the learned features. The main caveat is simply to keep the class distribution balanced enough that the abundant classes don't dominate training.
