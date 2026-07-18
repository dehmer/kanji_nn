Topic: Isotropic vs anisotropic Scaling
Bot: Gemini
Date: 26 07 09

> Concerning training LSTM model for classification of Japense Kana and Kanji characters based on vector data, what are the pros and cons of isotropic vs anisotropic scaling x and y to the glyphs bounding box?

When preparing vector data of Japanese Kana and Kanji characters for an LSTM classifier, you must normalize the coordinate points. The choice between isotropic (uniform) and anisotropic (non-uniform) scaling significantly affects how the LSTM learns stroke features.
## Direct Answer
Isotropic scaling preserves the character's original aspect ratio, which keeps critical structural shapes intact but leaves smaller or elongated characters with pad/blank space. Anisotropic scaling stretches every character to fill a uniform bounding box, maximizing feature resolution for complex characters but distorting distinctive aspect ratios (like the verticality of "川" versus the squareness of "口").
------------------------------
## Isotropic Scaling (Uniform)
This method scales the $x$ and $y$ coordinates by the same maximum factor, preserving the original proportions of the glyph.
## Pros

* Preserves Aspect Ratios: Vital for Japanese script where the width-to-height ratio distinguishes characters.
* Natural Stroke Angles: Directional vectors and angles remain mathematically accurate for the LSTM.
* Consistent Geometric Style: Hand-drawn styles maintain their natural spatial relationships.

## Cons

* Variable Density: Narrow characters (like "し" or "川") leave empty space in the bounding box.
* Loss of Fine Detail: Tiny, complex Kanji scaled down to fit a wide box lose stroke resolution.

------------------------------
## Anisotropic Scaling (Non-Uniform)
This method scales the $x$ and $y$ axes independently so that every glyph completely fills a standard square bounding box (e.g., $0$ to $1$ for both axes).
## Pros

* Maximum Feature Resolution: Maximizes the spatial workspace for complex, dense Kanji (like "鬱" or "驚").
* Consistent Boundary Conditions: Every sequence start and end point maps to a highly uniform bounding envelope.
* Reduces Size Variance: Normalizes the differences between inherently small Kana (like small "っ") and large Kanji. [1]

## Cons

* Severe Shape Distortion: Destroys structural identity; a tall, thin stroke sequence is stretched wide.
* Altered Angular Features: LSTM coordinate deltas ($\Delta x, \Delta y$) change, warping the learned representations of stroke directions.
* ** Kana Ambiguity**: Increases confusion between naturally similarly shaped characters that only differ by proportions.

------------------------------
## Impact on LSTM Architecture

* Sequence Dynamics: Anisotropic scaling alters the velocity and direction vectors between sequential points. If your LSTM relies on $\Delta x$ and $\Delta y$ inputs, anisotropic scaling introduces artificial directional shifts depending on the character's original bounding box.
* Feature Invariance: Isotropic scaling forces the LSTM to learn aspect-ratio invariance. Anisotropic scaling forces it to learn distortion invariance. For Japanese handwriting, aspect ratio is usually a highly informative prior.

------------------------------
To help determine the best approach for your specific dataset, could you tell me:

* What is the average sequence length (number of coordinate points) per character?
* Does your dataset contain mostly isolated, clean characters, or continuous cursive handwriting (Gyousho)?
* Are you feeding raw coordinates ($x, y$) or directional deltas ($\Delta x, \Delta y$) into the LSTM?


[1] [https://www.nature.com](https://www.nature.com/articles/s41598-021-81909-x)

> I'm feeding directional deltas on highly normalized characters (KanjiVG).

## Direct Answer
For highly normalized vector data like [KanjiVG](https://kanjivg.tagaini.net/), isotropic scaling is the superior choice because KanjiVG characters are already designed within a strict, uniform 109 × 109 square bounding box. Since you are feeding directional deltas (Δ x, Δ y) to your LSTM, isotropic scaling preserves the true angles and relative stroke lengths, whereas anisotropic scaling would introduce artificial distortions to the directional vectors of naturally narrow or wide characters.
------------------------------
## Why Isotropic Scaling Wins for KanjiVG + Deltas

```
[Original KanjiVG] ──> Already perfectly square (109x109 canvas)
   │
   ├─► Isotropic:   Maintains 1:1 aspect ratio ──► Angles ($ \theta $) & Deltas remain mathematically pure. ✅
   └─► Anisotropic: Stretches non-square glyphs ──► Distorts Deltas ($\Delta x \neq \Delta y$) & warps angles. ❌
```

## 1. Mathematical Purity of Deltas
An LSTM relying on directional deltas processes the sequence as a trajectory of vectors:
$$\Delta x_t = x_t - x_{t-1}, \quad \Delta y_t = y_t - y_{t-1}$$

* Isotropic: Keeps the scaling factor k identical for both axes ($k_x = k_y$). The resulting direction angle $\theta = \tan^{-1}(\Delta y / \Delta x)$ remains completely unchanged. Your LSTM learns clean, consistent stroke directions (e.g., a perfect 45° diagonal slash remains 45°).
* Anisotropic: Appears redundant because KanjiVG characters are already normalized to a uniform canvas. However, if applied to the individual glyph bounding boxes, it forces $k_x \neq k_y$. This warps the stroke angles dynamically based on the character's unique width-to-height ratio, making it significantly harder for the LSTM to generalize shared radicals across different Kanjis.

## 2. KanjiVG Character Design
KanjiVG is meticulously aligned to a standard square grid.

* Radical Consistency: Radicals (like the water radical 氵 or person radical 亻) appear on the left side of many characters with highly consistent geometric proportions. [1]
* The Flaw of Anisotropic Boxing: If you stretch a narrow character or a specific asymmetric character to force-fill a bounding box, the shared radicals will change shape, size, and angle. This strips away the structural consistency that makes KanjiVG ideal for machine learning.

## 3. Handling Radical Densities
Because KanjiVG is a clean typographic reference dataset rather than erratic human handwriting, it does not suffer from the "noisy stroke resolution" issues that normally justify anisotropic scaling. The LSTM will easily capture the micro-details of dense Kanji because the coordinates are crisp, digital, and noise-free.
------------------------------
To optimize your LSTM model for KanjiVG, I can help you with the following:

* Propose an ideal input feature vector combining deltas with stroke-state flags (pen up/pen down).
* Provide a Python preprocessing snippet to scale KanjiVG data isotropically.
* Recommend LSTM architectures (like Bidirectional LSTMs) tailored for vector-based stroke recognition.

Which of these steps would you like to explore next?

[1] [https://blog.lingodeer.com](https://blog.lingodeer.com/kanji-radicals/)

> Got it! Isotropic scaling is preferred for directional deltas. Another question is whether or not to scale (isotropically) against widget bounding box (or SVG view port) or the actual glyph's bounding box. Here I would go for the later, because KanjiVG and the widget drawing area make for different space around the actual character, right?

## Direct Answer
You should absolutely scale against the SVG viewport / canvas bounding box (the global 0 0 109 109 coordinate system) rather than the individual glyph’s tight bounding box.
While you are correct that KanjiVG files contain whitespace around the actual strokes, scaling tightly to the glyph’s specific boundaries would completely break the LSTM's ability to learn cross-character spatial consistencies.
------------------------------
## Why Scaling to the SVG Canvas Wins

[SVG Viewport Box (109x109)] ──► Keeps spatial context intact.
                                  "一" stays wide and flat at the center. ✅

[Glyph Bounding Box (Tight)]  ──► Destroys spatial context.
                                  "一" gets amplified into a massive horizontal wall. ❌

## 1. Preserving "Absolute" Structural Meaning
In Kanji, a radical's size relative to the standard block is a fundamental feature of the writing system.

* The "一" (ichi) problem: The character for "one" (一) is a single horizontal stroke. If you scale to its glyph bounding box, its height is nearly zero. Scaling it up to a standardized box would aggressively amplify tiny vertical hand tremors or control point noise into massive vertical deltas.
* The "口" (kuchi) vs "囗" (box radical) problem: The standalone kanji for mouth is "口" (small, sits in the center). The enclosure radical is "囗" (large, fills the outer edges, seen in "国"). If you scale tightly to the glyph bounding box, "口" and "囗" look mathematically identical to your LSTM. Scaling to the SVG viewport ensures the LSTM learns that "口" has a small bounding footprint and "囗" has a large one.

## 2. Consistency for Downstream App Inference (The "Widget" Reality)
If you build a feature where a user draws a character in a UI widget, they naturally write within a square frame.

* By training your LSTM on the SVG viewport coordinates, you align the training data's coordinate frame with the real-world drawing frame.
* If you train on tight glyph bounding boxes, you would be forced to calculate a tight bounding box on the user's live strokes before calculating deltas. If a user pauses mid-character or draws a single stroke, your tight bounding box shifts wildly with every added point, destabilizing your delta calculation in real time.

## 3. How to Handle the Padding
The whitespace padding embedded within KanjiVG's 109x109 grid is not "wasted space"—it is the literal margins required to keep characters balanced when displayed next to each other.
Instead of scaling to the glyph box, simply map the full SVG canvas directly to a normalized numeric range:

# Optimal normalization approachnormalized_x = x / 109.0normalized_y = y / 109.0

This forces all coordinates cleanly into a $[0.0, 1.0]$ space while perfectly maintaining absolute positions, stroke sizes, and whitespace integrity.
------------------------------
Since we've locked down the spatial normalization scheme, would you like to:

* Look at a Python example to parse the KanjiVG XML and extract these canvas-normalized deltas?
* Discuss how to handle stroke breaks (when the pen lifts) so the LSTM doesn't treat the leap to the next stroke as a massive continuous delta?

> OK, you're right about loosing spacial information. For example for small variants of や, ゆ,よ or つ.

## Direct Answer
Exactly. The small variants—like the small sokuon "っ" (tsu) or the small yōon "ゃ", "ゅ", "ょ"—are the perfect examples of why scaling to the glyph box fails.
If you normalized tightly to the glyph's bounding box, a small "っ" would be stretched and scaled up until it looks identical to a full-sized "つ". By keeping the coordinate system bound to the global SVG viewport, the LSTM can instantly tell them apart because the small versions have significantly shorter stroke lengths and occupy a much smaller spatial area within that 109 × 109 frame.
------------------------------
## Real-World Contrast: Small vs. Large Kana
To see exactly how this protects your LSTM's accuracy, consider what happens mathematically to your directional deltas under both systems:

| Feature / Character | Large "つ" vs. Small "っ" (Glyph Box Scaling) | Large "つ" vs. Small "っ" (SVG Viewport Scaling) |
|---|---|---|
| Delta Magnitudes (Δ x, Δ y) | Identical. The small version's deltas are artificially amplified to match the large version. | Different. The small version yields much smaller delta values, giving the LSTM an explicit size cue. |
| Spatial Position | Lost. Both characters are centered into the exact same normalized box. | Preserved. The small version stays tucked into its natural quadrant (usually the lower-left or center-right depending on layout). |
| LSTM Classification | High Ambiguity. The model must rely purely on hyper-subtle curvature differences. | Highly Distinct. The model easily differentiates them via sequence step sizes and total bounding footprint. |

## The "Pen Lift" Delta Warning
Now that you are committed to the global SVG viewport system, there is one critical sequence anomaly to watch out for when calculating your directional deltas.
When moving from the end of one stroke to the start of the next, the "pen lift" creates a massive jump in coordinates. If you feed that huge jump into the LSTM as a standard Δ x, Δ y, the model might mistake it for a massive, lightning-fast stroke.
To fix this, it is standard practice to append a pen-state flag ($p_t$) to your delta features:

* $\mathbf{x}_t = [ \Delta x_t, \Delta y_t, p_t ]$
* Where $p_t = 0$ if the pen is down (actively drawing a stroke), and $p_t = 1$ if the pen is up (the invisible leap between strokes).

------------------------------
Since you are working with the clean KanjiVG structure, how would you like to handle this data sequence?

* Do you need a vector preprocessing strategy to filter out or flag those invisible "pen up" jumps?
* Would you like to look at sequence padding methods (like masking) to handle the fact that KanjiVG characters vary wildly in stroke count?
