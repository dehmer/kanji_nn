# Handoff: Segment-Boundary Discontinuity in LF (Savgol) Smoothing

## Project context

LSTM/Transformer for single Japanese character classification from vector
(pen) stroke data, not raster. Reference geometry comes from KanjiVG.
Handwritten (hw) data captured via iPad + Apple Pencil (Flutter app),
5-column raw layout `[timestamp, x, y, pressure, pen-down/-up]`, xy
normalized to [0,1]. Strict stroke order enforced. Physical capture canvas
has been constant at 21x21mm for recent datasets, but the person explicitly
does not want size-based tuning — any solution should be robust to roughly
+/-75% scale variation, so canvas size is deliberately out of scope.

Three datasets: katakana (47), hiragana (46), kanji (~80, kanken-10 level),
~180 characters total.

## Core stance carried through the whole project (do not violate)

- We are **not** aligning/morphing hw strokes toward the KanjiVG reference.
  Positional offset, overall trajectory/curvature deviation from reference,
  and idiosyncratic hooks/corners are the stroke's **identity** and must be
  preserved.
- The reference (KanjiVG path, its Bezier segment classification, OBBs,
  handle geometry) may be used as a **diagnostic yardstick** — to decide
  *where* to trim, *where* to size a filter window, *which* local shape to
  expect — but must never supply actual output coordinates. Precedent
  already established for DTW-based trimming and for seeding curve-fit
  parameters via similarity transforms; same principle applies to
  segmentation-based smoothing.
- Working style: 80:20, iterative, eyeball-and-adjust over
  metric-perfection. No ground truth exists for "clean" hw strokes, so
  success is judged by overlay inspection plus, where useful, cheap
  quantitative diagnostics (not a trained/validated model).
- Person's background: 35+ yr SW engineer, ~2 months into Python, strong
  general CS/math-adjacent instincts, leans functional. Wants critical
  pushback, prose over code-dumps-first, no unsolicited code before it's
  been discussed. Enjoys pair-programming with snippets going back and
  forth. Does not like multiple-choice prompts; prefers prose explaining
  options.

## Named artifact taxonomy (agreed, stable — reuse this vocabulary)

1. **Staircase / micro-jitter** — one mechanism: sensor quantization noise.
   Manifests as visible flat "stairs" in slow regions, as generic small
   noise in fast regions. Same root cause, not two problems.
2. **Motor-jitter** — physiological tremor (~8-12Hz), a real signal (true
   hand motion), not a measurement artifact. Distinct from #1.
3. **Continuous wobble / "serpentine"** — longer-period oscillation around
   an intended straight/gently-curved line. Same underlying cause as #4
   (underdamped motor response, low glass friction removing haptic
   damping) but a different *form*: a frequency-band phenomenon, plausibly
   handled by smoothing.
4. **Motor-overshoot** — corner overshoot + course-correction. A discrete,
   localized event tied to a direction change, not a frequency-band thing.
   **Explicitly parked/deferred.** Do not pull this back into scope
   without the user raising it. One observed related symptom worth
   remembering: a tiny **loop** immediately before a hane in 青's last
   stroke (月 component) — flagged as a symptom of overshoot, not yet
   investigated further, still parked.

Quantization/tremor (#1/#2) are already handled well by a first,
narrow-window Savgol pass (`savgol_smooth_hf`, see below) — considered
settled/good.

## Pipeline architecture (existing, working code)

Functional-composition pipeline over `Stroke` objects (frozen dataclass,
`raw: np.ndarray [t,x,y,pressure]`, `.clone(features=..., props=...,
sticky=...)` to produce new immutable strokes without mutating history).
`compose(f1, f2, ..., fn)` runs **right-to-left** (`f1(f2(...fn(x)))`) —
important when reading pipeline lists, since dependencies must resolve in
actual execution order, not source order.

Existing modules (all previously reviewed, considered solid, not in scope
to redesign):

- `Character.of_npy` / `Character.strokes()` — loads raw npy, splits into
  per-stroke `Stroke` objects, attaches KanjiVG path (scaled via
  `transform.py`, `a=1/109`) as `sticky["path"]`.
- `arc_length.py` — computes raw + Gaussian-smoothed arc length features
  (`raw:s`, `gauss:xy`, etc). The Gaussian/turning-angle machinery here is
  a **leftover from an abandoned CPD/HMM trimming attempt** — available,
  not actively wired into current cleanup, don't assume it's load-bearing.
- `turning_angle.py` — signed turning angle off `gauss:xy`, same
  leftover-diagnostic status.
- `resample_path_equidistant.py` / `resample_equidistant.py` (in
  `kanji_nn.svg`) — resamples the KanjiVG reference path to arc-length-even
  points; each output point carries the **original path segment index**
  it came from (3rd column) — this segment-index tagging is the seed for
  everything in the segmentation work below. Deliberately fixed 50 samples
  per curve segment (2 for lines) regardless of `error` param — flagged as
  premature-optimization territory, explicitly not worth revisiting
  (person's call).
- `dtw_rle.py` / `trim_region.py` (current version) — DTW-based head/tail
  dwell trimming. `open_begin=True, open_end=True`, asymmetric step
  pattern. Detects "stagnation" runs (hw points that map to the same
  reference index repeatedly) at the very head/tail only (mid-stroke gaps
  intentionally not cut — that branch is commented out). **Current
  `trim_region.py` builds the new `Stroke` directly from sliced
  `stroke.raw`** (not via `reset.py`), which is what makes real
  timestamp/pressure survive into trimmed data — confirmed empirically
  (irregular ms-scale deltas printed post-trim, not synthetic indices).
  `reset.py` is likely now dead code in this pipeline — status not
  confirmed, worth asking if it's still used elsewhere.
  Trimming is considered **done/settled** — strokes may still be "shaky"
  right at surviving head/tail, that's expected and fine.
- Empirically confirmed data bounds across all 3 datasets: `n_points` range
  6–220 (shortest observed: 校 stroke 6, an n=6 case that also stressed the
  trimmer). This sets a hard floor for any window-based logic.

## Savgol smoothing — two-stage, chained (current working design)

Two independent functions (person deliberately chose **not** to unify them
into one parameterized function, for flexibility — window/polyorder logic
is expected to keep diverging):

- `savgol_smooth_hf` — first pass, narrow window, targets
  quantization+tremor. Writes `features["savgol:hf:xy"]`. Window derived
  from `n_points` (`k≈0.2`), floored/capped/forced-odd, `polyorder` fixed
  at 2. **Considered good, working well across all 3 datasets on
  whole-stroke application.**
- `savgol_smooth_lf` — second pass, wider window, targets serpentine
  wobble. Reads `features["savgol:hf:xy"]` as input (chained, not
  independent). Writes `features["savgol:lf:xy"]`.

**Whole-stroke lf worked great on katakana (47) and kanji (~80) — "nothing
to complain here."** Failed visibly on a handful of hiragana with long
strokes containing a **tight loop** embedded within an otherwise long
stroke (e.g. み stroke 1, む stroke 1, そ) — a single per-stroke window
sized off overall `n_points` is too wide relative to the loop's own small
arc-length extent, so the wide-window polynomial fit smooths through the
loop and visibly shrinks/rounds it, and often over-rounds a sharp corner
immediately adjacent to the loop too.

Root cause understood and agreed: **one fixed window per stroke can't
adapt to highly non-uniform local feature scale within that stroke.**

## Segmentation-based fix (in progress, current best approach)

Chosen direction: split each stroke into per-authored-Bezier-segment
pieces (using DTW alignment between `savgol:hf:xy` and the resampled
reference path, `open_begin=False, open_end=False` — different DTW config
than trimming's, since this is a full anchored correspondence, not an
open-ended excess-detection problem), then run `savgol_filter`
independently on each segment slice, with window/polyorder derived
per-segment rather than per-stroke, then `np.vstack` the pieces back
together.

Key implementation details already settled:

- **Segment-index labeling bug fixed**: reference point 0 (the zero-length
  `Move` segment) is relabeled to segment 1 *on the reference array itself*
  before lookup (`s = xys[:,-1].copy(); s[0] = 1; alignment =
  column_stack([W.index1, W.index2, s[W.index2]])`), not by patching row 0
  of the alignment output — robust even if more than one hw point maps to
  reference index 0 (only confirmed empirically to hit it once so far, but
  this version doesn't rely on that holding).
- **`window` per segment**: same `n`-based formula as before, but `n` is
  now the segment's own DTW-aligned point count (varies a lot per
  segment — observed range in one stroke: 3 to 49). Falls back to
  unfiltered `query[start:end]` when the computed window would exceed the
  segment's own `n` — this is considered *correct* behavior (a segment too
  short to filter is usually exactly the tight/small feature that
  shouldn't be aggressively smoothed anyway), not a gap to fix.
- **`polyorder` per segment**: originally derived from `window` size alone
  (`min(max_polyorder, window-1)`) — this was identified as **wrong**: at
  small windows this pins polyorder near the window ceiling, which
  destroys smoothing power exactly where it's most needed (near-interpolation,
  not smoothing) rather than being about window size being "too small"
  per se.
  **Corrected approach, now implemented**: `polyorder` is looked up
  directly from `classify_bezier`'s categorical label (already computed,
  reused from prior work): `near-straight → 1`, `left-bend/right-bend → 2`
  (default), `s-bend → 3`. Rationale: polyorder isn't a "how much
  smoothing" knob (that's `window`'s job) — it's a "what shape can even be
  represented" knob, and there's a natural, small, discrete mapping from
  Bezier shape category to minimal sufficient polynomial degree (quadratic
  can't represent an inflection, only cubic can — so s-bend specifically
  needs 3; everything monotonically curving or straight needs at most 2).
  Degree ≥4 not believed to be needed for any real KanjiVG segment shape.
- **A "second-difference norm" curviness proxy** (`second_diff_norm` in
  the current `dtw_segmentation.py`) was tried as a possible tightness
  signal / seed for adaptive window-sizing. **Rejected**: it is
  literally `|B''(0)/6|² + |B''(1)/6|²` for a cubic Bezier — a real,
  correct curvature-endpoint sample, but **not scale-invariant** (scales
  with `chord_length²`), so a big lazy bend can out-score a small genuine
  loop purely on absolute size. If revisited, must be normalized by
  `chord_length²` first, analogous to how `bezier_handle_geometry`
  normalizes handle magnitudes by chord length. Currently unused
  (`k_proxy` computed but not wired into `params()`).
  `bezier_obb`'s `ratio` (height/width in chord-aligned local space) and
  `bezier_handle_geometry`'s `magnitudes` (handle length / chord length)
  remain the more promising *already scale-invariant* tightness signals if
  a continuous (not just categorical) tightness measure is wanted later —
  not yet used for window-sizing, still on the table.

## Result of segment-based approach on む (test character), so far

- **Loop preserved correctly** — む stroke 1's loop no longer shrinks (a
  clear improvement over whole-stroke lf, which had visibly collapsed it).
  This is the main goal achieved.
- **A dent-like residual artifact in stroke 2, segment 0** (an
  OBB-ratio≈0.09 segment, `n=12`) partially resolved by moving from
  polyorder=3 (old window-derived rule) to polyorder=2 (new class-driven
  rule, since this segment classifies as `right-bend`, not
  `near-straight` as both people guessed/bet on beforehand — bet was
  **lost**, worth remembering not to trust intuition here over checking
  `classify_bezier` directly).
- **Segment-boundary discontinuities are real and visible** (confirmed by
  deliberately exaggerating `k` to 0.8 to make them obvious), specifically
  observed at two boundaries in む stroke 1:
  - **Boundary 0/1**: `polyorder` jumps 2→1 *and* `window` jumps
    dramatically (3→15, since segment 1 is a long near-straight run,
    n=30). Hypothesis: mismatched polyorder (different curve families —
    line vs quadratic — with no shared constraint forcing tangent
    agreement at the seam) is a real, distinct discontinuity mechanism.
  - **Boundary 3/4**: `polyorder` is **identical** on both sides (2, 2) —
    this boundary sits *inside* the tight-loop region (OBBs here are
    tightly clustered with very different orientations from the earlier
    OBB visualization). Discontinuity still shows clearly here despite
    matched polyorder. This **falsifies "polyorder-mismatch is the whole
    story"** — conclusion: **window-size mismatch + fast-changing local
    tangent direction (high OBB-orientation delta) between adjacent
    segments is at least as significant a discontinuity driver**, possibly
    more so, specifically in fast-turning/loop regions.
  - Net conclusion (both people agree): two distinct, evidenced
    discontinuity mechanisms exist. Polyorder-matching alone does **not**
    fully solve continuity. Some form of boundary blending/overlap is
    needed going forward — not because it was assumed necessary
    up-front, but because it's now empirically demonstrated.

## THE ACTUAL OPEN TASK FOR THE NEW SESSION

Design and implement **seam blending/stitching** between independently
Savgol-filtered segments, to eliminate the two confirmed discontinuity
mechanisms above, while preserving everything that currently works
(loop shape, hane sharpness, corner sharpness — all previously verified
good on 青, 字, 気, and now む).

Things to figure out together, in the spirit of measure-before-build /
80:20 that's driven this whole project:

1. **Should blending be universal or targeted?** Given boundaries with
   small window/polyorder/orientation deltas may already look fine
   (per earlier eyeballing on 青/字/気, before the loop-heavy hiragana
   cases surfaced problems), it may be wasteful to apply expensive
   overlap-and-discard everywhere. Consider computing a per-boundary
   "risk score" (window-size delta, polyorder delta, OBB-orientation
   delta between adjacent segments — all cheap, already-available
   quantities) and only blending where it exceeds some data-driven
   threshold, rather than guessing a threshold blind.
2. **What form should blending take?** Candidate: classic **overlap-and-
   discard** — extend each segment's filter window some amount into its
   neighbors before running Savgol, then keep only the central portion of
   each result when stitching, so no segment's boundary values come from a
   filter with a hard, no-neighbor array edge. Alternative, cheaper:
   simple linear (or cosine) cross-fade of the two independently-filtered
   segments' values over a small blending zone straddling the boundary —
   doesn't fix the underlying fits' disagreement in tangent, just
   visually papers over it. Worth discussing tradeoffs concretely rather
   than picking blind.
3. **How to actually measure success this time?** The person's own
   qualitative eyeballing has been reliable and sufficient throughout this
   project (no ground truth exists, and that's accepted) — but for this
   specific problem, a cheap quantitative diagnostic (e.g. position jump
   and/or tangent-angle jump at each boundary, before vs after blending)
   was discussed as worth adding, specifically to confirm whether
   blending measurably reduces the two confirmed discontinuity
   mechanisms, not to replace eyeballing but to complement it.
4. Keep in mind the still-parked but-related observation: the tiny loop
   before 青's final hane. Not in scope to solve now, but worth watching
   whether the eventual blending design has any bearing on it (probably
   not, since that's a motor-overshoot/topological issue, not a
   segmentation-seam issue — but flag if it turns out related).

## Style/process reminders for the new session

- Prose over code-first; only write code when explicitly asked, and keep
  it as one function/snippet at a time unless told otherwise, matching
  the existing pipeline's style (`stroke.clone(features=..., props=...)`,
  small pure functions).
- Push back with real technical substance when something doesn't hold up
  (the person explicitly wants this and has course-corrected the
  assistant more than once this session, productively).
- Don't reach for the reference (KanjiVG) to supply output coordinates —
  only as a yardstick for decisions (segmentation, window/polyorder
  choice). This boundary has been explicit and consistent throughout.
- Iterative, eyeball-driven, 80:20 — no need for perfect or fully general
  solutions before moving on.