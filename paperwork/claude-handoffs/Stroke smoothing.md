# Handoff: Handwriting Stroke Smoothing Pipeline

## Goal

Preprocess handwritten kanji/kana stroke samples (raw touch/stylus captures,
`[t, x, y, pressure, pen]`) so they're "pleasant to look at" — genuine motor
jitter and glass-tablet noise removed, while corners, curves, and the
sample's own displacement/length/shape are preserved. This is deliberately
**decoupled** from the longer-term goal (bringing hw samples closer to the
distribution the LSTM/transformer classifier was augmented on) — that
rework is still pending, so for now we're optimizing for a clean, honest
denoised stroke, not distribution-matching.

## Hard constraint (do not violate)

**The KanjiVG reference is a classifier, never a fit target.** It may be
used to *label* points/segments (straight vs. curved, corner vs. not), but
must never pull a raw point's coordinates toward the reference's geometry.
Stroke displacement and arc length are downstream-critical and must reflect
the handwriting sample itself, not an idealized version of it. This
constraint survived the whole session and the recent decision to lean on
authored SVG geometry (see below) does *not* relax it — richer reference
data only makes labeling more precise, it changes nothing about the
never-snap rule.

## Where we ended up (the fork in the road)

Two options were on the table for how corner/segment classification gets
its ground truth:

1. **Hard road**: derive segment/corner labels from scratch via curvature
   or RDP on the (sampled-only) `wkb` reference polyline.
2. **Easy road (chosen)**: use the actual authored KanjiVG SVG path data
   (cubic Béziers) as the reference geometry. User can supply parsed paths
   via `svg.path` / `parse_path`. This gives exact segment boundaries and
   analytic curve shape, no curvature-threshold guessing.

**Decision: going with the easy road.** Open item: clarify whether `wkb`
(currently used by `vg_trace_align`/DTW) is a lossy derivative of this SVG
data or a separate asset — determines whether existing `wkb`-based DTW
alignment can be reused as-is or needs to be redone against the Bézier
paths directly.

## Pipeline architecture facts (established this session)

- `compose(...)` in `character.py`'s pipeline runs **right-to-left** —
  actual execution order is bottom-up from how it reads.
- Trimming (`trim_region`) already runs late in that order, and
  `Stroke.trim()` wipes `features`/`props` to `{}`. Confirmed: there is
  **no structural requirement** that it must — user is open to letting
  specific data ride through trim, e.g. as a column parallel to `raw`
  (sliced congruently via the same `region[0]:region[1]`), rather than
  via `features`/`props` (which get discarded).
- `vg_trace_align` (DTW against `wkb` reference) currently runs *before*
  `trim_region`, and its `path` correspondence array is the natural hook
  for reference-informed per-point labeling — but it does **not** survive
  trimming today. If reused, it needs to be carried as a per-point column
  alongside `raw`, not a `features`/`props` entry.
- `Character.strokes(smooth_fn=...)` used to call `smooth_fn(raw[:,1:3])`
  (xy only). **This constraint is now moot** — confirmed the eventual
  output format is plain X/Y per stroke (and per character) written to
  disk; `t`/`pressure` are not part of the output shape and don't need to
  be reconciled with anything. A separate pipeline stage now strips
  timestamp/pressure from `raw` early (user-added).
- `dtw_align.py` is an O(N·M) double Python loop — fine at current stroke
  lengths, flagged as a future perf concern if reused post-trim or on
  longer strokes.
- `curvature.py` has a documented, unfixed failure mode: near-zero
  `gauss:ds` (jitter with little net displacement) blows up curvature via
  division by near-zero arc length. Relevant if curvature is ever computed
  directly on raw/lightly-smoothed points.
- Augmentation (`s_weighted_random_walk_noise` etc.) already adds noise in
  **delta/relative space**, not absolute — direct answer to an earlier open
  question. Noise is pace-weighted: more noise on RDP-thinned/straight
  segments, less on corner/curve-dense regions — i.e. training data already
  encodes "corners and curves stay clean, straight runs get perturbed."
  This is currently **not** being pursued (decoupled from this session's
  goal) but is relevant context for later distribution-matching work.

## Current working pipeline (built this session)

Operates on already-trimmed, `t`/`pressure`-stripped strokes:

1. **`resampling_uniform(stroke, n_out=None)`** — resamples `xy` to
   uniform arc-length spacing via `np.interp` against `stroke.features['raw:s']`.
   Returns a **new `Stroke`** built directly (not via `.clone()`, since
   point count changes and old derived features would go stale — same
   reasoning as why `trim()` wipes features). `n_out` defaults to original
   point count; explicit **fixed spacing** (rather than fixed point count)
   was discussed as the more principled choice for cross-stroke-consistent
   corner-detection windows, but user deferred that — current code still
   defaults to preserving original point count per stroke. Easy to swap
   later. ゑ flagged as a good stress-test candidate (longest single-stroke
   arc length) if/when spacing consistency is revisited.

2. **`gauss_1d(stroke, sigma=1.0, mode="reflect")`** — Gaussian smoothing
   of resampled `xy`, stored as `stroke.features["gauss:xy"]` (does not
   overwrite `raw`/`xy`). `mode="reflect"` confirmed correct default
   (matches existing `arc_length.py` convention; better than `nearest`
   at preserving tangent direction near stroke endpoints, `wrap` is wrong
   for open strokes).

3. **`turning_angle(stroke, w=3)`** — windowed turning-angle (not
   instantaneous curvature): heading looking forward minus heading looking
   backward over a `w`-point window, wrapped into `(-π, π]`. Operates on
   `stroke.features["gauss:xy"]`. Returns angles only — **no
   corner/straight labeling yet**; that's a deliberate deferral so
   thresholds can be picked by eyeballing real data first. Endpoints
   (`w` points on each side) are `NaN`. Vectorized via array slicing, no
   Python loop.

### Sigma tuning (in progress, real data)

Tested on 虫 stroke 1 (long flat run + sharp hook) at σ = 1.0, 1.5, 2.0,
3.0, 5.0:
- Real corner peak (~1.15 rad at σ=1.0) shrinks only gradually with σ
  (1.09 → 1.01 → 0.85 by σ=3.0; still present even at σ=5.0).
- A small motor-noise bump (glass-tablet overshoot, ~t=22, originally
  ~0.05 rad) shrinks faster and is essentially gone into the noise floor
  by σ=2–3.
- **Currently defaulting to σ=3.0** as a working value. Explicitly flagged
  as *not yet validated* against shorter/sharper strokes where corner vs.
  noise separation may be tighter — the 虫 hook stroke had unusually
  generous headroom (noise ~0.2, corner ~1.15) that may not generalize.

### Known open problems in the angle signal (seen on ね stroke 1, a
self-crossing loop with a hane)

- **Sign-wrap ambiguity near ±π**: a near-180° reversal can land on either
  side of the wrap boundary depending on tiny sampling differences,
  producing e.g. −3.1 rad where a geometrically similar event elsewhere
  reads +1.9 rad. **Fix for corner *detection*: threshold on `abs(angle)`**,
  not signed value. Not yet implemented.
- **Intentional reversal vs. unintentional self-crossing overshoot,
  co-located at the same extreme-angle event**: user wants to preserve the
  former (e.g. a hane's sharp direction change) while removing the latter
  (motor overshoot causing a brief self-crossing loop). Proposed approach
  (not yet implemented): a **separate, local spatial self-proximity check**
  — does the pen's position shortly after the event come back close (in
  space, over a short arc-length window) to its position shortly before?
  A pure reversal doesn't re-approach its recent path; an overshoot-with-
  correction does, briefly. This is a second pass on top of turning-angle,
  not a replacement for it.
- **Sustained moderate angle over long spans (not a spike) on genuinely
  curved segments** (e.g. ~0.3–0.9 rad over t≈100–175 on the ね loop):
  plausibly just what smooth curvature looks like in this signal, not
  noise and not a corner — a single global magnitude threshold tuned on
  simple strokes (like 虫) may not transfer to loop-shaped strokes. Not
  yet resolved.
- **General observation, not yet acted on**: even at σ=3.0, both straight
  and curved sections of ね still look visibly wobbly. This is expected/
  correct at this stage — Gaussian smoothing here is deliberately light,
  just enough to make turning-angle reliable, not a final denoising pass.
  Residual wobble is meant to be absorbed later by segment-aware curve
  fitting (line fit on straight runs, spline fit on curved runs), once
  segmentation exists to tell the fitting step which to apply where.

## Immediate next steps (in likely order)

1. Get parsed SVG path data (`svg.path`/`parse_path`, cubic Béziers) into
   the pipeline as the reference geometry. Clarify relationship to
   existing `wkb` asset (lossy derivative vs. separate).
2. Re-parameterize Bézier segments by arc length (not raw Bézier `t`,
   which doesn't advance at constant speed along a curve) so they're
   comparable to the raw stroke's own `s`.
3. Rebuild (or redo) DTW correspondence against this richer reference to
   get exact segment-type (straight/curved) and boundary labels per
   reference position, then propagate onto raw stroke points via the
   correspondence — same "reference as lookup table for labels" shape
   discussed for the `wkb` polyline case, just with exact ground truth
   instead of curvature-derived estimates.
4. Decide how/whether DTW correspondence (or derived per-point labels)
   should ride through `trim_region` as a `raw`-parallel column.
5. Implement `abs(angle)` thresholding for corner detection; validate
   threshold across multiple strokes (simple line+bend, tight loop with
   hane, smooth curve) rather than tuning on one example.
6. Implement the local self-proximity check to separate intentional sharp
   reversals from self-crossing overshoot artifacts.
7. Once segmentation/labels exist: implement segment-wise fitting (line
   fit for straight runs, spline fit for curved runs) as the actual
   final denoising step — this is where the still-visible residual
   wobble is expected to get resolved, not before.

## User working style notes

- 35+ years SW engineering experience; new to Python specifically —
  flag idiom/gotchas (mutability, numpy slicing/broadcasting, dataclass
  semantics), skip design hand-holding.
- Prefers simple/practical solutions; wary of "consistency" as a
  standalone justification when it might paper over a real, specific
  requirement worth understanding on its own terms.
- Iterates in small, concrete steps with real plotted data at each stage
  rather than large speculative leaps — sigma/threshold choices are being
  decided empirically per-example, not analytically upfront.
- Wants code only when explicitly asked; otherwise discussion/review.