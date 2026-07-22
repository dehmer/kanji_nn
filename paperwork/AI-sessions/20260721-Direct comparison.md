Topic: Direct comparison
Bot: Gemini/ChatGPT
Date: 26 07 21

> I want to quickly check if there is something to the following idea.
In a broader context I want to clean-up (digital ink) Kanji stroke data acquired by hand-writing with Apple pencil on iPad. For a couple of well understood reasons, this data contains unwanted, dirty artefacts especially in the head and tail regions of the stroke. Now since writing was done with a corresponding KanjiVG path image as backdrop/reference for the writer, correct stroke order and high similarity to the original stroke (checked by DTW error threshold) are guaranteed. How would you assess the chances that "direct comparison" between sample strokes and original KanjiVG path data could be used to reliably detect and trim excess artefact at head and tail? Was this done before? Do you have any suggestion on how to approach this?

Gemini:

The short answer is yes, your chances of success are exceptionally high, and this is a solid engineering approach.
Because you have already established a constrained pipeline where correct stroke order and high similarity to the reference are guaranteed (via DTW error thresholds), you have bypassed the hardest problems in handwriting recognition (such as structural segmentation and alignment ambiguity).
------------------------------
## Has This Been Done Before?
Yes, but typically as an unromantic, "in-house" preprocessing step.
In digital ink literature (such as early Pen Computing research or biometric signature verification frameworks), removing hooks, "pigtails," and contact artifacts at the boundaries of strokes is a well-known requirement.
Most published papers focus on writer-independent recognition where templates are unavailable. However, in template-matching scenarios or handwriting-teaching apps (where a backdrop reference like [KanjiVG](https://github.com/KanjiVG/kanjivg/releases) is actively used), performing boundary trimming using the alignment path of Dynamic Time Warping (DTW) is a standard, highly reliable trick. [1, 2]
------------------------------
## Why Your Setup Makes This Easy
The artifacts you are dealing with—often caused by the physical behavior of the Apple Pencil hitting or lifting off the glass before or after the intentional gesture—manifest as rapid, erratic changes in velocity, micro-loops, or short, trailing tails that diverge wildly from the smooth cubic Bézier curves defined in KanjiVG.
Because you are already computing the DTW cost matrix, the solution is already hiding in your data.
------------------------------
## Step-by-Step Implementation Approach
To isolate and trim these artifacts using the KanjiVG reference, implement the following pipeline:
## 1. Analyze the DTW Alignment Path (The "Warping Path")
When you align your sample stroke $S$ (length $N$) to the KanjiVG reference stroke $R$ (length $M$), DTW produces an optimal warping path:
$$W = w_1, w_2, \dots, w_K \quad \text{where } w_k = (i_k, j_k)$$
Here, $i_k$ corresponds to the index in your sample stroke, and $j_k$ corresponds to the index in the KanjiVG reference.

* Detecting Head Artifacts: Look at the beginning of the warping path. If the first few points of the KanjiVG stroke ($j = 0, 1, 2$) map to a long sequence of indices at the beginning of your sample stroke ($i = 0, 1, \dots, p$), it means the writer spent time moving the pen around before actually progressing along the true trajectory of the stroke.
* Detecting Tail Artifacts: Look at the end of the warping path. If the final indices of the sample stroke ($i = q, \dots, N$) map almost entirely to the single final point of the KanjiVG stroke ($j = M$), those points represent an intentional or unintentional exit trail/hook.

## 2. Apply Boundary Trimming Thresholds
Do not just drop points blindly. Use the DTW localized distance matrix to find the exact cutoff points:

* The Trim Rule: Trim the head up to index $p$ and the tail from index $q$ where the individual, local distance/cost between $S(i)$ and $R(j)$ drops below a stabilized threshold.
* The Speed/Curvature Filter: Validate the trim by checking kinematic features. Artifacts typically exhibit a sudden spike in curvature (turning sharply as the pen leaves the glass) or a massive drop/surge in velocity. If the DTW indicates an alignment lag at the boundary and it correlates with an unnatural kinematic anomaly, slice it off.

## 3. Handle Scale and Sampling Densities
KanjiVG curves are smooth vectors. Your raw iPad data is likely highly sampled (often 120Hz to 240Hz depending on the iPad model).

* Equidistant Resampling: Before running DTW, resample both the KanjiVG stroke and the iPad stroke so that points are spaced at equal spatial distances (e.g., every 2 pixels), rather than equal time steps. This removes velocity bias from the spatial alignment and makes head/tail detection strictly geometric.

------------------------------
## Potential Gotchas to Look Out For

* Intentional Calligraphic Features: In Japanese calligraphy (Shodo), features like the sharp brush-up hook (Hane) or the deliberate stop/flair (Tome) can look like artifacts to a naive geometric trimmer. Ensure your thresholding accounts for the stroke type—KanjiVG files do not explicitly flag stroke types natively, but certain radicals always demand a Hane.
* Over-trimming: If a user writes a stroke perfectly but starts a tiny bit late, a naive DTW trim might shave off the valid start of the stroke. Ensure your trimming window is restricted to the first and last ~10–15% of the total stroke length.

## Proactive Next Steps
If you want to refine this algorithm further, it would be helpful to understand a bit more about your current data pipeline:

* What programming language or framework are you using to compute your DTW matrices (e.g., Python's fastdtw, Swift, JavaScript)?
* Are you saving or utilizing the pressure or tilt data provided by the Apple Pencil? (Pressure drops to near-zero during lifting tails, which makes artifact detection flawless).


[1] [https://github.com](https://github.com/KanjiVG/kanjivg/releases)
[2] [https://spire.ee.iisc.ac.in](https://spire.ee.iisc.ac.in/src/talks.php)

ChatGPT:
