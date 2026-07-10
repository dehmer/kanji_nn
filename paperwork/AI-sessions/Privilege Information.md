Topic: Component Composition
Bot: ChatGPT
Date: 26 07 07

> Hey there! Quick basic question about RNN/LSTM classification models. Say I encoded structural information about Kanji composition from KanjiVG's path groups somehow in a usable form for training. Hand-written test samples for inference clearly cannot provide this information. Is it possible or even common that training tensors differ in width from tensors for inference? Luckily you are a machine and I am not too embarrassed to ask, haha.

Not embarrassing at all—this is actually a very important question, and it gets at a fundamental property of machine learning models.

The short answer is: **no, not in the way you're describing.** The model expects the same feature representation during training and inference. If your training tensor has an extra "KanjiVG structural information" component that is absent at inference time, you're effectively asking the model to solve two different problems.

For example, suppose your training input is

```
[x, y, pressure, stroke_order, KanjiVG_structure]
```

but at inference you only have

```
[x, y, pressure, stroke_order]
```

The learned weights connecting to `KanjiVG_structure` have nowhere to apply during inference. The input dimensionality simply doesn't match.

### What *can* differ?

There are a few situations that sound similar but are actually fine.

**Sequence length can differ.**

RNNs and LSTMs are specifically designed for variable-length sequences:

```
Training:
50 × 12 features
73 × 12 features
31 × 12 features

Inference:
65 × 12 features
42 × 12 features
```

Only the **number of timesteps** changes. The feature dimension (12 here) stays fixed.

---

**Batch size can differ.**

Training:

```
batch = 64
```

Inference:

```
batch = 1
```

This is completely normal.

---

**Masked or padded inputs are common.**

You can pad sequences to a maximum length during training but use different lengths during inference, as long as the padding/masking scheme is consistent.

### Your KanjiVG idea

This is actually an interesting problem because you're talking about **privileged information**.

During training, you know things that a human writing on a tablet doesn't reveal:

* radical hierarchy
* enclosing components
* path groups
* semantic decomposition
* exact SVG grouping

During inference, you only see pen trajectories.

This is a well-known setting in machine learning called **Learning Using Privileged Information (LUPI)**, introduced by Vladimir Vapnik.

The basic idea is:

> Use extra information during training to build a better internal representation, but don't require it during inference.

There are several ways to accomplish this.

### Option 1: Teacher–student (knowledge distillation)

Train a large teacher model:

```
trajectory
+
KanjiVG structure
      ↓
 Teacher
```

Then train a student model that only receives trajectories:

```
trajectory
      ↓
 Student
```

The student learns to imitate the teacher's predictions.

This is probably the most common approach.

---

### Option 2: Auxiliary prediction

Feed only trajectory data into the encoder.

During training, ask it to predict both:

* the Kanji identity
* the KanjiVG structural encoding

```
trajectory
      ↓
   LSTM
   /   \
class  structure
```

At inference, you simply discard the structure prediction head.

The network has learned representations that encode structural concepts even though it never receives structure as input.

This approach is extremely common.

---

### Option 3: Multi-task learning

Closely related to the above.

Loss =

```
classification loss
+
λ × structure reconstruction loss
```

Again, no structural input is needed during inference.

---

### Option 4: Pretraining

Use KanjiVG to pretrain a representation of character structure, then fine-tune on handwriting.

This has become a popular strategy in deep learning.

### Something even more interesting

Depending on how you encode KanjiVG, you might not need to feed it as an input feature at all.

Suppose each Kanji has a structural graph like

```
⿰
├─ 木
└─ 目
```

or

```
left-right
   ├── radical A
   └── radical B
```

Instead of adding this graph to the input tensor, you could use it only to define an additional loss during training.

For example:

```
Handwriting
     ↓
   Encoder
     ↓
 Embedding
    ↙     ↘
class    predict KanjiVG embedding
```

At test time, only the left branch is used.

This is often a cleaner design because the inference pipeline stays identical to the training pipeline.

### My suggestion

If your end goal is handwritten Kanji recognition, I would **not** append KanjiVG structural features directly to the input tensor if those features won't exist at inference time.

Instead, I'd use KanjiVG as **supervision rather than input**. That means:

* Input: handwriting sequence only (during both training and inference).
* Targets during training: the correct Kanji label plus one or more structural targets derived from KanjiVG.
* Inference: handwriting only.

That lets the model learn structural regularities without depending on information it won't have when deployed.

For handwritten character recognition in particular, this is a very natural fit because the structural information acts like an "expert annotation" that shapes the latent representation, rather than becoming a required part of the observation.

> Wow! How a small and simple idea can morph into a major rabbit hole. Nevertheless, good explanation. I keep 'privilege information' and LUPI in mind. The proposals for now exceed my mental capacities (and ML knowledge) by far. But I also guess that's nothing they teach you at ML 101, haha.

Haha, that's a very common experience in ML.

A thought process often goes something like:

> "I'll just add this one extra feature..."
>
> ...which turns into...
>
> "Actually, this is a special case of a research area with papers going back 20 years."

LUPI is definitely **not** ML 101 material. A typical progression is more like:

1. Linear regression
2. Logistic regression
3. Neural networks
4. CNNs/RNNs/LSTMs
5. Transformers
6. Regularization, optimization, embeddings...
7. Then much later: multi-task learning, knowledge distillation, metric learning, domain adaptation, LUPI, self-supervised learning, etc.

So the fact that you stumbled into LUPI by thinking carefully about your data is actually a good sign. It means you're asking the kinds of questions that arise when moving from toy examples to real-world problems.

I also think you're asking the right question for Kanji recognition. The distinction between **what exists during training** and **what exists when the system is actually used** is one of the first practical design constraints people run into.

For your specific project, I'd actually recommend *not* trying to implement LUPI or teacher–student methods yet. They're powerful, but they also introduce quite a bit of additional machinery. I'd instead build the simplest possible pipeline first:

* Input: pen trajectory (coordinates, timing, pressure if available, pen-up/pen-down).
* Model: an LSTM (or bidirectional LSTM) classifier.
* Output: Kanji class.

Get that working well first. Once you have a baseline, then you can ask, "How can KanjiVG help?" At that point, you'll have something concrete to compare against, and you'll be able to tell whether the added complexity actually improves performance.

One thing I found particularly interesting from your question is that you weren't trying to "cheat" by giving the model extra information—you immediately realized that the information wouldn't exist at inference time. That's exactly the kind of consistency check that's easy to overlook when you're excited about a new feature, and it's a habit that pays off in ML projects.

And don't worry about ideas like auxiliary losses or multi-task learning feeling abstract right now. Once you've trained a straightforward classifier, those concepts become much easier to understand because they're essentially saying, "While you're learning the main task, also learn this related task from the same internal representation." It's a much smaller conceptual leap when you've already built the basic model yourself.

> Yeah, currently I'm on a good path to get the basics right. Just wanted to quickly check this hunch out.

I think you're approaching it exactly the right way.

One thing I've learned (and seen countless researchers recommend) is that **complexity compounds much faster than you expect** in ML. Every additional idea—a new loss term, another input modality, an auxiliary task—creates more moving parts that can fail or interact in unexpected ways.

A solid progression for your project could look something like this:

* ✅ Get the data pipeline right.
* ✅ Verify your trajectory representation.
* ✅ Train a basic LSTM/BiLSTM classifier.
* ✅ Make sure you can overfit a tiny subset (a classic sanity check).
* ✅ Get decent validation accuracy.
* **Only then** ask: "Can structural information improve this?"

That way, if an experiment fails, you know *what* changed.

I actually liked your original question because it came from reasoning about the deployment scenario rather than trying to chase accuracy. That's a healthy instinct. A lot of beginners (and, frankly, plenty of experienced people) accidentally build models that depend on information they'll never have in production. You caught that before writing a single line of code.

And here's another encouraging thought: your Kanji project is actually a nice domain for learning ML because the domain itself provides intuition. When you think "these two characters share a left radical" or "this enclosing component should matter," you're already thinking in terms of **inductive bias**—what prior structure you want the model to exploit. That's a much richer way to think than simply throwing a bigger network at the problem.

So I'd keep doing exactly what you're doing: when a hunch pops into your head, spend 15 minutes checking whether it's fundamentally sound, then put it on the shelf if it isn't immediately useful. Those little detours build a mental map of the field, even if you don't use the idea right away.

And don't be surprised if, six months from now, you look back at "privileged information" and think, "Oh... that's all they meant?" A lot of ML concepts feel impossibly abstract until you've built one or two complete systems. Then they suddenly become very concrete because you can point to a place in your own code and say, "That's where this idea would fit."
