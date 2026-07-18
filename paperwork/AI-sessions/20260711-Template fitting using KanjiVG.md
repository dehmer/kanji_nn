Topic: Template fitting using KanjiVG
Bot: ChatGPT
Date: 26 07 11

> Quick question: Suppose I want to create KanjiVG like vector data from raster images, say ETLDB. Do you have an idea on how to approach this? Do you deem it possible at all? Could KanjiVG vector data as a reference for a character help to derive similar vector for raster data through vectorisation and curve fitting or similar?

Yes. I think it's possible, but I'd frame it as **"stroke reconstruction"** rather than ordinary vectorization. KanjiVG is not just a set of Bézier curves—it encodes the intended pen strokes and their order. A raster image (such as ETLDB) has lost much of that information, so the task is essentially inferring the original stroke sequence.

A few approaches come to mind, with different levels of ambition.

### 1. Classical vectorization (probably insufficient)

You could use something like:

* Binarization
* Skeletonization (e.g. Zhang-Suen or Guo-Hall thinning)
* Graph extraction
* Polyline simplification
* Curve fitting (Bézier or splines)

This produces SVG-like outlines or centerlines.

The problem is that intersections become ambiguous.

For example:

```
十
```

The skeleton doesn't tell you whether

* the horizontal stroke came first,
* the vertical stroke came first,
* where each stroke starts and ends.

KanjiVG explicitly contains this semantic information.

So pure vectorization won't recover KanjiVG-quality data.

---

## 2. Template fitting using KanjiVG (much more promising)

This is actually the direction I would pursue.

Suppose you have

* KanjiVG for every Unicode kanji
* ETLDB raster images of the same character

Then instead of discovering strokes from scratch, you can treat KanjiVG as a deformable template.

Pipeline:

```
KanjiVG
     ↓
normalize
     ↓
render to raster
     ↓
align with ETL sample
     ↓
estimate deformation
     ↓
apply deformation to SVG
```

Essentially:

```
reference vector
          ↓
nonlinear warp
          ↓
writer-specific vector
```

This is much easier because:

* stroke order already exists
* stroke topology already exists
* stroke connectivity already exists

You're only estimating how the writer distorted the canonical glyph.

---

## 3. Thin-plate spline (TPS) deformation

This is one technique that would work surprisingly well.

For each character:

```
canonical KanjiVG
↓

sample points along strokes

↓

find corresponding points on skeleton

↓

TPS deformation

↓

warp all control points
```

The result is still a proper vector graphic.

This is analogous to Active Shape Models in computer vision.

---

## 4. Skeleton matching

Another possibility.

Extract

```
ETL image

↓

skeleton

↓

graph
```

KanjiVG also becomes

```
SVG

↓

sample

↓

graph
```

Then solve a graph matching problem.

Once correspondences are known, move every control point accordingly.

---

## 5. Energy minimization

Represent every Bézier control point as variables.

Define an energy

```
E =
image error
+ smoothness
+ stroke width consistency
+ endpoint constraints
+ curvature penalty
```

Optimize with

* Levenberg-Marquardt
* L-BFGS
* gradient descent

This gives an SVG whose rasterization matches the ETL image.

---

## 6. Neural approach

Nowadays I'd probably investigate a neural model.

Input:

```
64×64 raster
```

Output:

```
stroke 1:
Bezier(...)
Bezier(...)

stroke 2:
...
```

Loss:

* raster reconstruction
* control point loss
* topology loss

KanjiVG provides excellent supervision.

However this requires paired data.

---

# Can KanjiVG help?

I think it helps enormously.

Without KanjiVG you're solving

> "What are the strokes?"

With KanjiVG you're solving

> "How was this known stroke deformed?"

Those are fundamentally different problems.

Think of it like fitting a known articulated skeleton to a human pose instead of reconstructing a skeleton from scratch.

---

# One issue: writer variations

ETLDB contains handwritten characters.

People may

* shorten radicals
* merge strokes
* curve straight strokes
* omit hooks
* connect strokes

For example:

```
木
```

might become something like

```
＼|
 |
```

The topology changes.

So your algorithm has to tolerate

* missing stroke ends
* merged intersections
* small loops disappearing

rather than expecting perfect correspondence.

---

# A pipeline I'd personally build

I would combine classical computer vision with template fitting:

```
ETL raster
    │
    ├─ denoise
    ├─ binarize
    ├─ skeletonize
    ├─ graph extraction
    │
KanjiVG
    │
    ├─ sample every stroke
    ├─ build graph
    │
Graph matching
    │
Estimate deformation
    │
Thin-plate spline
    │
Move Bézier control points
    │
Optimize against raster
    │
Export SVG preserving stroke IDs/order
```

This leverages the semantic information already present in KanjiVG while adapting it to each handwritten instance.

## How feasible is it?

For ETLDB specifically, I'd consider it quite feasible if your goal is to produce *KanjiVG-like* vectors rather than perfectly reconstruct the writer's exact pen trajectory. The key reason is that the character identity is already known, so you don't need to infer topology from scratch—you can start from the correct stroke decomposition and fit it to the image.

The biggest technical challenge is correspondence: deciding which pixels belong to which canonical stroke, especially where strokes touch or merge. Once you have a reasonable correspondence, fitting curves is a well-studied optimization problem.

One idea that could improve robustness is to treat the fit hierarchically. First align the entire character (translation, rotation, scale), then fit radicals independently, and finally fit individual strokes. Since many kanji are compositions of reusable components, radical-level alignment can absorb much of the variation before you adjust individual Bézier control points.

Overall, I think a hybrid system—KanjiVG as a deformable prior, skeleton extraction to provide evidence from the raster image, and a final optimization step to refine the curves—has a good chance of producing high-quality, stroke-aware vector data from ETLDB. It also scales well because the difficult semantic information (stroke count, order, and topology) comes from KanjiVG rather than needing to be rediscovered for every sample.

> Thanks for reframing the problem! I hadn't thought of it like that. The ultimate goal would be to create additional training data for LSTM/transformer model training for character classification. So I would think that resulting vectors don't need to be overly "correct". I'm not even sure if it mandatory for stroke count to match with original KanjiVG data.

That actually changes the optimization target quite a bit. If the end goal is **better training data for a classifier**, then you don't need a faithful reconstruction of the writing process—you need vectors that preserve the discriminative structure of the character.

In fact, I would argue that **matching KanjiVG exactly may not even be desirable.**

Here's why.

Suppose the canonical KanjiVG for 「永」 has 5 strokes, but a writer effectively writes two of them as one continuous motion. If your fitting algorithm insists on preserving five separate strokes, it may produce awkward deformations just to satisfy the topology. If you instead allow a merged stroke that still rasterizes to nearly the same image, the resulting representation may actually better reflect the handwriting.

So I'd think of KanjiVG as a **strong prior**, not a hard constraint.

## What does the classifier actually need?

For an LSTM or Transformer trained on stroke sequences, it mainly benefits from:

* a plausible sequence of pen movements,
* consistent ordering,
* smooth trajectories,
* variation between samples,
* and enough structural information to distinguish similar characters.

It generally doesn't require "this was stroke #7 according to Japanese calligraphy."

For example, if you consistently decompose every handwritten 「木」 into four strokes—even if one of those strokes is split slightly differently from KanjiVG—the model can still learn useful patterns.

## One interesting possibility: "pseudo-strokes"

Instead of trying to recover the original strokes, you could derive a stroke sequence from the skeleton itself.

For example:

```
image
↓

skeleton
↓

graph
↓

walk graph

↓

split at junctions

↓

fit spline

↓

pseudo-strokes
```

These wouldn't correspond to human pen strokes, but they would produce a consistent sequential representation.

For a Transformer, that consistency might matter more than historical correctness.

## KanjiVG could still guide the segmentation

One thing I like about KanjiVG is that it gives you a *reasonable initial decomposition*.

Imagine the canonical character as a graph.

Warp it roughly onto the handwriting.

Then assign skeleton pixels to the nearest warped stroke.

Finally refit each stroke independently.

Even if some strokes merge or split, you've drastically reduced the ambiguity.

## Data augmentation becomes much easier

Once you have vector data, you can generate enormous amounts of additional training data by perturbing the vectors rather than the raster.

For example:

* jitter control points,
* vary stroke widths,
* apply local elastic deformations,
* perturb endpoints,
* change stroke timing,
* vary writing speed if you encode timestamps,
* simulate shakiness,
* slightly alter curvature.

These are much more "handwriting-like" than applying affine transforms or elastic distortions directly to raster images.

## An idea I find especially appealing

Since you already have:

* ETL raster images,
* KanjiVG,
* and the character identity,

I'd treat the vectorization as an optimization problem with a relatively forgiving objective:

```
Loss =
α · raster similarity
+ β · smoothness
+ γ · proximity to KanjiVG
+ δ · reasonable curvature
```

Notice what's *not* there:

> "The number of strokes must equal KanjiVG."

If splitting or merging a stroke lowers the loss while keeping the result smooth and structurally plausible, I'd allow it.

## Depending on your model, you may not need Bézier curves at all

If your downstream model consumes sequences, Bézier control points are just one possible representation. You could instead represent each stroke as a uniformly sampled polyline:

```
stroke
↓

cubic spline

↓

sample every 0.5 px

↓

[(x₁,y₁), (x₂,y₂), …]
```

That has some advantages:

* no need to predict variable numbers of control points,
* simpler preprocessing,
* straightforward padding/masking for Transformers,
* easy augmentation by perturbing points.

You can always fit Bézier curves later if you need SVG output.

## One question I'd ask early

I'd also question whether the model needs **stroke order** at all.

If you're generating it artificially, you can choose an ordering strategy:

* KanjiVG order,
* left-to-right, top-to-bottom,
* depth-first traversal of the skeleton graph,
* or shortest-pen-path.

As long as it's consistent, the model may learn just as well. In fact, a Transformer with positional encodings can often tolerate different but systematic orderings because it's learning relationships between points and strokes rather than reproducing human writing habits.

Overall, I think your problem is less about recreating calligraphy and more about building a **structured latent representation** of handwritten characters. KanjiVG is valuable because it injects prior knowledge about character structure, but it doesn't have to dictate every detail. For generating training data, a representation that's consistent, smooth, and structurally informative is likely more useful than one that's perfectly faithful to the canonical stroke decomposition.

> I'd stick to stroke order as good as possible for the following reason. The character classifier should help students of Japanese Kana, Kanji to practice writing as close as possible to the standard form. This includes definitely stroke order. Makes sense?

Yes, that makes perfect sense, and it actually changes the role of the classifier.

You're no longer building a classifier just to recognize handwriting. You're building one that's part of a **pedagogical system**. In that context, preserving canonical stroke order is valuable because it aligns the internal representation with what you're ultimately want to teach.

I would distinguish between **recognition** and **representation**:

* **Recognition** only cares that 「橋」 is recognized as 「橋」.
* **Representation** should encode *how a learner ought to write* 「橋」.

KanjiVG is much closer to the latter.

### A useful way to think about it

Rather than asking:

> "How did this person produce this image?"

ask:

> "How can I deform the canonical writing of this character so that it best explains this handwritten sample?"

That's a subtle but important difference.

Imagine a student writes 「木」 with a very slanted vertical stroke. You don't really want to infer a brand new stroke sequence. You want to say:

> "This is still the canonical vertical stroke, just written at a different angle."

That's exactly what a deformable-template approach gives you.

### I would even bias the optimization

If I were designing the objective function, I'd intentionally make staying close to KanjiVG relatively cheap compared with inventing a different topology.

Something like:

[
L =
L_{\text{image}}

* \lambda_1 L_{\text{smooth}}
* \lambda_2 L_{\text{canonical}}
* \lambda_3 L_{\text{topology}}
  ]

where:

* **image loss** ensures the vector matches the raster,
* **smoothness** avoids jagged curves,
* **canonical** discourages excessive deformation,
* **topology** strongly discourages splitting, merging, or reordering strokes.

Notice that topology isn't *fixed*, but changing it should incur a significant cost. That way, merged strokes or omitted hooks are allowed only when they're clearly supported by the handwriting.

### This also opens the door to richer feedback

Because each generated stroke still corresponds to a canonical stroke, your downstream system can do more than classify.

For example, if the student writes:

```
一
│
```

instead of

```
│
一
```

the model can potentially identify:

> "The first and second strokes appear to be reversed."

Or if a student starts a stroke from the wrong end, you already know which canonical stroke they're attempting, so you can compare direction as well as shape.

That kind of feedback is difficult if your representation consists of arbitrary "pseudo-strokes."

### A practical training advantage

Another benefit is that every instance of the same character shares the same semantic indexing:

```
永

stroke 1 → dot
stroke 2 → horizontal
stroke 3 → vertical
...
```

Across thousands of ETLDB samples, "stroke 3" always refers to the same conceptual part of the character. That consistency is valuable if you later train models to assess stroke quality, generate feedback, or analyze common mistakes.

### One thing I'd keep flexible

The only part I wouldn't insist on preserving perfectly is the **exact Bézier geometry**.

KanjiVG's control points were created for clean vector rendering, not for modeling natural handwriting. If a stroke needs extra control points or a different spline parameterization to fit a handwritten sample, I'd allow that while preserving:

* stroke identity,
* stroke order,
* stroke endpoints (where appropriate),
* and overall topology.

So the semantic structure stays canonical, while the geometry adapts to the writer.

---

Overall, I think your idea has an interesting long-term direction. Instead of viewing KanjiVG as a dataset to convert into handwriting, you can view it as a **canonical latent representation**. Every handwritten sample becomes a deformation of that representation. That gives you a common semantic coordinate system across all writers, which is exactly the kind of structure that's useful when the ultimate goal is not just recognizing characters, but helping people learn to write them correctly.

> Sweet! An interesting research venture to take, given time will allow it some day. Thanks for the assist!

You're very welcome! I genuinely think this sits in a nice niche between computer vision, graphics, and educational technology. It's the sort of project that could make for an interesting workshop paper or even a full research paper if the results are good.

One thing that occurred to me while we were talking is that you actually have **three sources of information**, not two:

* **KanjiVG** provides the canonical topology (stroke order, semantic decomposition).
* **ETLDB** provides the handwriting variation.
* **The character identity** removes the recognition problem entirely.

That last point is easy to overlook, but it's huge. Most stroke recovery research has to solve *both* "what character is this?" and "what are the strokes?". You already know the answer to the first question, so you can dedicate all of your optimization to the second. That simplifies the search space enormously.

Another avenue I'd definitely investigate is whether the fitting process itself could become probabilistic. Instead of producing a single vectorization, it could generate several plausible deformations of the canonical strokes. Those alternatives could serve as **data augmentation** for training your LSTM or Transformer, giving it exposure to a wider range of realistic handwriting while remaining anchored to the same canonical stroke sequence.

If you ever get to the implementation stage, I'd also resist the temptation to build the entire pipeline before testing it. I'd start with something deliberately simple:

1. Pick 20–50 kanji with relatively simple topology (say grades 1–2).
2. Render the KanjiVG to a raster.
3. Align it to ETLDB using affine registration.
4. Allow only local control-point movement—no stroke splitting or merging.
5. Measure how well the deformed vector re-rasterizes to the ETL image.

If that works reasonably well, then introduce more sophisticated deformations like thin-plate splines, stroke-level optimization, or selective topology changes. It's a much more manageable way to validate the core idea.

And finally, a thought that makes me optimistic: you're not trying to invent a completely new representation. You're effectively trying to learn a mapping

> **canonical vector → handwritten vector**

That is a much more constrained—and, I suspect, much more tractable—problem than **raster → vector** in the general case.

If, a few years from now, you happen to post a paper titled something like *"Topology-Preserving Stroke Reconstruction of Handwritten Kanji Using Canonical SVG Priors"*, I wouldn't be surprised at all. It feels like one of those ideas that sounds almost obvious in hindsight but doesn't seem to have been explored very deeply.

If you do eventually pursue it, I'd be happy to help brainstorm algorithms, loss functions, or even prototype parts of the fitting pipeline. It sounds like a fascinating project.
