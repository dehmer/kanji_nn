# Kanji/Kana Stroke Post-Processing — Session Handoff

## Project context
Cleaning up Kanji/Kana stroke vector data captured via a Flutter app on iPad
(Apple Pencil), sent as raw `PointerEvent` data to a Python web service. Post-processing
happens offline against NumPy saves. Guiding philosophy throughout: **as simple as
possible, not as good as possible.**

## Raw data format
- **Columns (5-col layout):** `[t, x, y, pressure, state]`
  - Some uploads use a 7-col layout: `[t, x, y, pressure, orientation, tilt, state]`.
    Orientation/tilt are dropped (reported as "boring"/near-constant, deprioritized).
  - `t`: ms, **character-relative**, starts at 0 for the first point of a character,
    increases monotonically across all strokes in that character.
  - `x, y`: isotropically normalized to widget bounds, range `[0, 1]`.
  - `pressure`: normalized to reported min/max from the API. "Normal" writing maxes
    out well below 1.0 — observed stroke plateaus ranged from ~0.14 to ~0.35 across
    different strokes/characters. **Not a fixed absolute scale across strokes.**
  - `state`: pen-down = `1.0`, pen-up = `0.0`.
- **Sampling rate:** ~8–14ms between samples, fairly regular, no major dropped-event gaps observed.
- **One `.npy` file = one whole character**, containing all strokes concatenated (raw
  Flutter capture). NOT one file per stroke (this was a source of early confusion —
  resolved).
- **Stroke boundary rule (important, verified against real data):** a stroke consists
  of a `state==1` run **plus exactly one trailing `state==0` sample** — the true
  pen-up point, with real `(x, y)` coordinates and typically (not always) `pressure≈0`.
  Splitting on `state==1` runs alone truncates every stroke by one point.
- Pressure at pen-down is **not** reliably 0 — `PointerDownEvent` can already report
  mid-ramp pressure. Don't use "pressure == 0" as a proxy for "true first contact."

## Known artifacts in the raw trajectories
1. **Micro-jitter** — low priority, not a major concern per the project owner.
2. **Start hooks/loops** — small noise squiggles right after pen-down, before the
   "real" stroke direction establishes.
3. **End hooks/loops** — noise after the intended stroke content is done, sometimes
   immediately following a *wanted* feature (e.g. a hane/はね hook), separated from it
   by a short straight run. Wanted and unwanted features can look geometrically similar
   — this is NOT solvable by simple start/end point trimming.
4. **Axis-aligned "staircase" quantization** — confirmed in multiple unrelated
   characters (ロ, ナ). Diagonal motion gets captured as discrete x-only/y-only unit
   steps rather than smooth diagonal movement. This is a **real property of the raw
   coordinate data**, not a measurement/parsing bug on our side. Empirical check
   (one example) showed staircase segments run at roughly half the speed of diagonal
   segments — consistent with the general "Apple Pencil prediction/coalescing +
   quantization becomes visible at low speed" explanation for this class of behavior
   on iPad/Flutter (informational, not verified against Flutter source/docs).

## Hypotheses tested this session
- ❌ **Absolute pressure threshold marks the artifact cut point.** Disproven — a
  "clean" stroke with a much lower pressure ceiling (~0.14 vs ~0.35) still needed the
  same relative reasoning; a counter-example was found where pressure hadn't declined
  at all near a still-uncertain end region.
- 🤔 **Relative pressure-decline-onset (slope change, not absolute value) marks the
  artifact cut point.** Two supporting examples so far (セ hane case, ナ backtrack
  case) — both show a "knee" from gentle decline to steep collapse right around where
  a manual cut made sense. **Not yet validated against clean examples** (i.e., do
  clean stroke ends show a single smooth decline with no knee?). This is exactly the
  kind of check the "clean enough" reference pile (in progress) is meant to enable.
- 🤔 **Direction reversal near the end, independent of pressure.** Confirmed real (not
  a screenshot-zoom illusion) in the ナ example — genuine backtrack in x after the
  intended stroke end, co-occurring with the pressure knee above. Reversal + relative
  pressure-decline-onset as a **joint signal** is the current best lead for end-artifact
  detection, still needs testing against more (especially clean) examples.
- Pressure in general has been explicitly **deprioritized** — "last resort if
  everything else fails," per the project owner.

## Bugs found & fixed (or diagnosed) this session

### 1. Original screener script (`stroke_artifact_screener.py`, first iteration)
- `max_turn_deg` saturates at 180° by construction (wrapped angle) — hit in 74% of
  starts across a real batch, not discriminating.
- `cum_turn_deg` (sum of absolute per-sample turning angle) is dominated by
  **micro-jitter and quantization staircase**, not real hooks — confirmed via a
  top-ranked "candidate" (ロ, stroke 2) that was geometrically a clean straight line.
  Root cause: exact-duplicate consecutive points cause `atan2(0,0)` degenerate
  headings, injecting spurious ~180° turns on both sides of every duplicate.
- **Dedup alone is insufficient** — removing exact duplicates still leaves real
  axis-aligned staircase zigzag (833° cum_turn even after dedup, vs. 2813° raw).
- **Gaussian smoothing (sigma≈1.0 sample) on x/y before computing heading/turn is the
  better fix** — reduced the same false positive's cum_turn to ~250–300°, and a
  synthetic-corner test showed monotonic real bends are preserved essentially 100%
  regardless of sigma (0.9–2.0 tested), because smoothing only cancels *oscillatory*
  turning (jitter/staircase), not one-directional bends. `mode='nearest'` preferred
  over `mode='reflect'`/`'mirror'` for boundary handling, to avoid mirroring real edge
  geometry back on itself.
- `primary_direction` (PCA/SVD on the stroke's middle 30–70% by arc length) breaks
  down for stroke shapes with **no dominant straight axis** — confirmed via the
  SVD's own singular-value ratio: ~30+ for genuinely straight-cored strokes (ロ
  stroke 2, ぬ stroke 0) vs. ~2–3 for curvy/loopy strokes (の, ぬ's looped stroke).
  **Proposed fix (not yet implemented):** expose this ratio and use it to gate how
  much `max_dev_from_primary_deg` contributes to the suspicion score — full weight
  above some threshold, reduced/zero below it. Threshold not yet calibrated; needs a
  broader dataset (the ratio-distribution stat request below is aimed at this).

### 2. Refactored pipeline (`Stroke` dataclass + composed metric functions)
- `split_strokes` — **verified correct**: does include the trailing pen-up sample
  (confirmed against ナ real data). One latent fragility: it drops the *last* column
  unconditionally (`raw[:, :-1]`) rather than the column actually named by the `pen`
  parameter — works today only because `pen` happens to equal the last column index.
  Suggested fix: `np.delete(raw, pen, axis=1)`.
- `Stroke`'s docstring/comment claims a 5-column raw layout (`[t,x,y,pressure,state]`)
  but `split_strokes` strips `state` before a `Stroke` is built, so `stroke.raw` is
  actually 4 columns. Behavior is correct/intentional (state isn't needed per isolated
  stroke); comment is just stale.
- **`metric_speed` (original version) doesn't measure speed.** `dx/ds, dy/ds` (arc-length
  derivative) is the unit tangent vector by definition — `hypot` of it is ≈1
  everywhere for any curve, confirmed on real data (mean 0.992, std 0.029), regardless
  of actual hand velocity. Fixed by splitting into:
  - `metric_tangent`: arc-length-based `tx, ty` (feeds curvature — this needs to stay
    arc-length-based).
  - `metric_speed`: **time-based** `dx/dt, dy/dt` — genuine writing speed.
- **Regression introduced during that split:** `metric_curvature` still divided
  `tx/ty` by `features["speed"]`, which after the split now holds the *time-based*
  value (~0.002 typical) instead of the old (coincidentally ~1) arc-length one.
  Confirmed this corrupts curvature by ~8 orders of magnitude (real numbers: correct
  K ≈ -3.9..135, corrupted K ≈ -7.6e8..1.5e9). **Fix:** don't renormalize `tx/ty` at
  all in `metric_curvature` — they're already the unit tangent by construction. If
  defensive renormalization is wanted, divide by `hypot(tx, ty)` (their own
  magnitude), never by an unrelated externally-computed feature.
- `Stroke.clone()`'s duplicate-feature-key check (built earlier this session) will
  correctly throw if two different metric functions both try to write the same
  feature name (e.g. old `metric_mystery` and new `metric_speed` both writing
  `"speed"`) within the same composed pipeline — validated this is a real, useful
  safety net, not just theoretical.

### 3. "Mystery" → renamed **"straightness"**
- What was originally the buggy "speed" calculation (`hypot(dx/ds, dy/ds)`) turns out
  to be a genuinely useful, different metric once correctly understood: for three
  consecutive points A→B→C, it equals `|A→C| / (|A→B| + |B→C|)` — the ratio of the
  direct shortcut distance to the actual two-hop path length through B. Verified
  exact match (to 4 decimal places) against real data.
  - Reads exactly **1.0** wherever the path is locally straight (no bend at B).
  - Drops **below 1.0** in proportion to how sharply the path bends at B (triangle
    inequality) — approaching 0 for a near-180° reversal.
  - **Bounded to (0, 1]** by construction — does not have the unbounded-blowup
    problem that `cum_turn_deg` had with jitter/staircase accumulation. Likely a
    better building block for a rewritten edge-suspicion score.
  - **Known limitation:** always reads exactly 1.0 at the very first and last sample
    of a stroke — this is a tautology of the one-sided finite-difference formula at
    boundaries (displacement magnitude divided by itself), not a real measurement.
    Hasn't caused problems yet since real artifact bends have appeared a few samples
    in from the true edge in every example seen so far, but worth remembering.

## Current architecture decisions
- `Stroke` is a **frozen dataclass**: `raw: np.ndarray` (canonical, 4-col
  `[t,x,y,pressure]` post-split), `features: dict[str, np.ndarray]` (per-point,
  growable), `props: dict` (scalar/non-per-point, e.g. would hold something like the
  SVD ratio). Named `@property` accessors (`.t`, `.x`, `.y`, `.xy`, `.pressure`) on
  the raw side, so access is consistent whether a value lives in `raw` or `features`.
- `Stroke.clone(features=None, props=None, force=False)` — shallow-copy update via
  `dataclasses.replace`, checks for duplicate keys (raises unless `force=True`) and
  validates new feature arrays match `n_points` in length. Immutable-update pattern,
  matches the project owner's FP leanings.
- `Character.of_npy()` loader handles both 5-col and 7-col layouts (drops
  orientation/tilt from 7-col).
- `Character.strokes(smooth_fn=identity)` — pluggable smoothing function applied to
  `(x, y)` jointly at stroke-construction time. Current known trade-off: whichever
  `smooth_fn` is passed **replaces** `raw`'s x/y — there's no single `Stroke` object
  with both raw and smoothed coordinates side by side. Workaround if ever needed: call
  `strokes()` twice (once with `identity`, once with the smoothing fn) and merge via
  `clone()`.
- Metric functions are unary `Stroke -> Stroke` transforms, chained via `compose()`
  (rightmost-first application, standard math convention) inside `process_file`:
  `arc_length -> speed/tangent -> curvature`.
- Smoothing: `gaussian_filter1d(xy, sigma=1.0, axis=0, mode='nearest')` applied to
  x/y before any kinematic derivative is computed. `mode='nearest'` chosen over
  `'reflect'`/`'mirror'` deliberately (see bugs section above).
- `metric_arc_length` adds a tiny per-index epsilon ramp (`arange(n) * 1e-12`) to the
  cumulative arc length to guarantee strict monotonicity, avoiding divide-by-zero in
  `np.gradient` when duplicate points remain. Side effect: this makes it impossible to
  detect *true* zero-length segments from `s` after this transform — duplicate/near-
  duplicate detection should be done on raw `ds` before this epsilon is added, if ever
  needed again (e.g. for a staircase-severity stat).

## Data collection in progress (project owner's task, across katakana/hiragana/kanji)
Originally requested:
- min arc length (over all strokes)
- pressure: max and std (per stroke)
- stroke duration (ms): min, max, std

Additionally suggested (to resolve open questions above), not yet confirmed collected:
- **Singular-value ratio (elongation/curviness) distribution, per stroke** — to
  calibrate the `primary_direction` gating threshold.
- **Staircase/quantization severity per stroke** — e.g. fraction of near-zero-length
  or purely axis-aligned segments — to see how widespread this is and whether a
  single fixed smoothing sigma is enough everywhere.
- **Min points per stroke** — very short strokes may have `EDGE_FRACTION`-based edge
  zones overlapping the whole stroke, making start/end scoring meaningless for them.
- **Min/max strokes per character** — sanity check on stroke-splitting logic at the
  high end (many-stroke kanji).

## Open items / natural next steps
1. Finish refactor (in progress by project owner) using the `Stroke`/`compose()`
   architecture and the corrected `metric_curvature`.
2. Rebuild the candidate scanner/scoring using **`straightness`** as the core signal
   instead of the old unbounded `cum_turn_deg`, now that it's understood and named.
3. Gate `primary_direction`/`max_dev_from_primary_deg` by the SVD singular-value
   ratio once enough data exists to pick a threshold.
4. Use the growing "clean enough" reference pile to actually test the
   reversal + relative-pressure-decline-onset joint hypothesis for end artifacts
   (and find/confirm an analogous start-side signal — not yet identified).
5. (Low priority, parked) Velocity-adaptive smoothing — vary sigma by local speed
   instead of a single fixed value, motivated by the staircase-severity-vs-speed
   correlation found late in this session. Suggested lightweight version if revisited:
   two-tier sigma (higher below a speed threshold) rather than a continuous function,
   in keeping with the project's simplicity preference.