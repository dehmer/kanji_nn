# Kanji/Kana Stroke Post-Processing — Session 2 Handoff

Continuation of `Stroke_Post-Processing.md` (session 1). This document covers
everything since. Read session 1 first if starting fresh — it's still the
source of truth for raw data format, known artifacts, and the `Stroke`/
`compose()` architecture.

## Companion file
`cuts_final_clean.csv` (95 rows) — the full manually-logged cut-point dataset
from this session. Columns: `literal, code_point, stroke_idx, duration,
head_cut, tail_cut, tail_region`. `tail_region = duration - tail_cut`.
`tail_cut` empty = no reversal found on that stroke's tail (a "clean ending").
One transcription error in the original paste (字 stroke 0) is already fixed
in this file — see "Manual cut-point dataset" below for what it was.

## 1-tier data collection (early in session)
- Built `dataset_data_points.py`: walks a dataset dir, loads `.npy`, splits
  strokes via `kanji_nn.conditioning.split_strokes`, tags each per-point
  DataFrame with `dataset`/`char`/`stroke_idx`, then reduces to one
  1-tier summary row per stroke (`n_points`, `duration_ms`, `arc_length`,
  `pressure_max`, `pressure_std`). Outputs `stroke_summary.csv` (per-stroke)
  and `stroke_summary_by_dataset.csv` (`groupby(dataset).describe()`).
- Ran across all 3 datasets: **657 strokes** total (106 hiragana_48, 151
  katakana_49, 400 kanken-10_80).
- **Pressure non-fixed-scale confirmed** at full scale: `pressure_max` ranges
  0.144–0.689 across strokes (mean 0.41, std 0.08).
- **Dataset-level stroke differences**: hiragana strokes are ~2x longer
  (duration, arc_length, n_points) than kanken kanji strokes on average.
- **`avg_speed = arc_length / duration_ms`** added as a free derived 1-tier
  stat (no new npy pass needed, pure ratio of existing columns). Result:
  mean speed is *nearly identical* across all 3 datasets (~0.001 units/ms,
  within ~10%). This **corrects** the "hiragana strokes are slower"
  interpretation above — they're not slower, they just cover more absolute
  distance at about the same drawing speed.

## Dakuten/handakuten discovery
Investigating the global min-arc-length / min-n_points / min-duration
outliers (2nd percentile thresholds: arc_length 0.173, n_points 28.12,
duration 226ms) led to katakana ベ and ビ. Both extremes turned out to be
the **dakuten (゛) diacritic strokes** — tiny (23–37 point), geometrically
disconnected marks in the upper-right of the glyph, captured as real
physical pen strokes in this dataset rather than a fixed annotation.
Handakuten (゜) strokes are the same phenomenon (small circles). **Decision:
strip both from the datasets for now** rather than build diacritic-aware
handling — whatever point-count/bbox heuristic identifies them today would
likely double as a diacritic detector later if ever revisited.

## Manual cut-point investigation — the 5-region stroke model
Proposed and adopted:
1. pen-down phase
2. established initial trajectory
3. stable body (may contain hane, self-intersection — legitimate complexity)
4. final trajectory
5. pen-up phase

**Cutting rule**: only regions 1 and 5 are ever eligible to be cut. Regions
2 and 4 are measurement/reference-only (e.g. establishing a direction or
pressure baseline to compare against) and are never trimmed themselves.
This structurally prevents any detector from ever trimming into the stable
body, even on a false positive.

Region 5 (pen-up) is extended backward by a **fixed duration** to absorb
"dirty air" before the true pen-up point; region 1 (pen-down) is understood
to extend forward from the first sample the same way, symmetric mechanism
(both anchored on a real, well-defined sample — first/last point — even
though the *pressure* at those anchors isn't reliably informative).

**Window-size risk, checked against real data**: fixed windows of 100ms per
side fully consume (leave zero stable body) 6 of the 90 shortest strokes in
`stroke_summary.csv`; 40–60ms per side consumes none. A hard floor (skip
cutting entirely if the cuttable zones would consume the whole stroke) is
necessary, not a rare edge case — it's a real outcome at the short end of
the distribution. This check only accounted for regions 1+5; real
consumption is higher once 2/4 are added on top.

**Manual cut-point dataset (final, 95 strokes, 13 kanken-10_80 kanji)**:
- `head_cut` present in **100%** of strokes (95/95) — every stroke shows a
  head-side reversal signature.
- `tail_cut` present in **89%** (85/95) — 10 strokes have genuinely clean
  endings (足, 字×3, 学×3, 校, 空, 赤, each contributing to that count — see
  companion CSV for exact list).
- Overall mean `head_cut` ≈ 103ms, mean `tail_region` ≈ 61ms — ratio ~1.7:1,
  confirming the `||1|| > ||5||` hunch, and by a wide margin.
- Per-character `tail_region` means range from 30ms (字) to 93ms (足), with
  a rough two-cluster split (低: 字/校/学/文/空/左, 高: 村/赤/音/玉/貝/森/足).
  **Caution**: checked this against per-character SEM — the extremes are
  real (well outside combined error bars), but middle-of-pack ordering is
  not reliable at n=3–12 strokes per character; several adjacent characters
  are statistically indistinguishable. A "森 is my nemesis, causes anxiety"
  hypothesis was floated (tension → sloppier pen-up) and briefly looked
  like a clean single-outlier story on 7 characters, but **fell apart once
  all 13 were in** — 森 became one of six similarly-elevated characters
  (足 is actually higher), not the sole extreme. A candidate "session
  fatigue/drift" explanation (head_cut flat across the session, tail_region
  rising) was also floated and **ruled out** — logging order was random,
  not chronological, so the apparent ordering effect was a code-point-sort
  artifact, not a real trend. **Conclusion reached**: stopped chasing
  per-character explanations deliberately — project owner's call, "endless
  unexpected influences on writing ergonomics," prefers a rule set that
  holds in the majority of cases over explaining every variation.
- Physiological/hardware research consulted (see below) supports the shape
  of the region model in general (ballistic-core / corrective-edges is a
  known general property of aimed movements — Woodworth's two-component
  model — not specific to this dataset) but was **not** used to import any
  specific numbers; real data took priority throughout.

### Grounding research consulted (web search, cited in-chat)
- Fitts' Law / speed-accuracy tradeoff; signal-dependent motor noise
  (control-signal-scaled noise as explanation).
- Woodworth's two-component model of aimed movement (primary ballistic +
  secondary corrective submovement) — maps directly onto the 5-region idea.
- Isochrony principle (Viviani & Terzuolo) — duration relatively invariant
  across amplitude *for the same gesture at different sizes*. Explicitly
  **not** the same experiment as our own cross-dataset avg_speed finding;
  flagged as a related-but-distinct concept, not used to explain it.
- Duration-vs-amplitude slope as a genuine/forged signature discriminator —
  suggestive for a "psychological, rapid stroke-switching" angle, no direct
  citation found for that specific claim.
- Tablet/stylus friction studies (Alamargot & Morin 2015; Denier van der Gon
  & Thuring 1965): low friction reduces proprioceptive feedback, disturbs
  trajectory calculation even in skilled writers, and adults compensate by
  **increasing pen pressure** on low-friction surfaces — a good independent
  reason pressure was already deprioritized in session 1's pipeline (it may
  partly encode a surface-compensation strategy, not pure natural dynamics).

## `straightness` — deep dive this session
Formula unchanged from session 1: `|A→C| / (|A→B| + |B→C|)` for consecutive
points A,B,C. Two genuinely new, load-bearing findings:

1. **Raw is more discriminating than smoothed, empirically, not just by
   hunch.** Gaussian smoothing (even sigma≈1.0) measurably blunts real sharp
   corners, not just noise — demonstrated on ベ's real zigzag peak
   (straightness minimum moved from a sharp raw value up toward 0.96 after
   smoothing). Project owner independently confirmed via visual comparison
   ("even small smoothing σ>0.1 can entirely even out an otherwise perfect
   spike/dent") and found that curvature-peak/straightness-minimum
   alignment, clean on raw data, **shifts left under smoothing** — plausibly
   because Gaussian smoothing is symmetric in *index* space but real strokes
   aren't sampled at constant speed (asymmetric deceleration into vs.
   acceleration out of a real corner is itself grounded in the
   two-component-model literature above). **Decision: stick with raw data**
   for cut-point work going forward. My own attempt to independently verify
   the curvature-shift claim hit the classic duplicate-point
   zero-arc-length degeneracy (needs the epsilon-ramp guard from session 1)
   — inconclusive, not a refutation.
2. **Strided straightness** (spacing A/B/C `k` samples apart instead of
   immediate neighbors) was tested as a smoothing-free alternative for
   damping jitter. Result on real data (ベ's zigzag peak vs. two known
   staircase false-positives): **inconsistent** — raw+strided values are
   non-monotonic in `k` and at `k=3` the false positive actually read as
   *more* suspicious than the real peak. Striding alone doesn't cleanly
   separate signal from staircase; it needs smoothing underneath it, which
   somewhat undercuts its original appeal as a way to avoid smoothing.

### The duplicate-point / dwell-cluster finding (important, still open)
Real strokes frequently contain runs of **exact duplicate points** right
after pen-down (confirmed: a literal 5-sample stationary dwell in 字 stroke
0, `[0.5182, 0.0927]` repeated). This produces two distinct degenerate
`straightness` outcomes depending on which two of A/B/C coincide:
- `A=B=C` (interior of a duplicate run) → `NaN` (0/0).
- `A=C`, B different (a genuine notch/fold-back) → exactly `0.0`.
- `A=B` or `B=C` only (entering/exiting a duplicate run) → exactly `1.0`
  (trivially "straight," **not** 0 — this surprised both of us).

**Open, unresolved question**: checked the *exact* logged `head_cut`/
`tail_cut` ms values from the manual dataset against raw immediate-neighbor
`straightness` and found they read `1.0`/`NaN` there, essentially never a
literal `0.0` — apparently contradicting an earlier informal observation
("straightness was 0 for almost all cuts"). Most likely explanation: the
logged cut ms is the project owner's *chosen practical cut location*, not
necessarily the exact sample where straightness bottoms out (cuts were
explicitly framed as "candidates near the final location," not ground
truth). **Deferred by project owner until after full manual logging was
done** — worth revisiting now that it is.

## `local_straightness` — new metric, added (not a replacement)
Reimplementation of the same ratio via a precomputed `stroke.features['s']`
(arc length) instead of raw x/y differences, plus `np.pad(..., mode='edge')`
for full N-length output. Three real differences from `straightness`, found
by direct comparison on 字 stroke 0's known dwell cluster:
1. **Regression**: `np.where(arc==0, 1e-6, arc)` turns the `NaN` dwell-
   interior case into a literal `0.0` — i.e., for *this* implementation,
   duplicate points genuinely do produce a spurious "reversal" reading,
   confirming the project owner's suspicion (raised, then reconciled) about
   duplicate points fooling the metric. Not yet fixed as of last check.
2. Boundary convention changed silently: old unpadded version had a
   documented tautological `1.0` at the true first/last sample; new
   edge-padded version just copies the nearest computed neighbor instead
   (arguably better, but a different, now-undocumented convention).
3. New execution-order dependency on `stroke.features['s']` already
   existing — no assertion guards this.

**Interaction discovered when combining both signals in edge detection**:
`straightness`'s `NaN` at a dwell boundary makes `diff` also `NaN`, and
`abs(NaN) > threshold` is always `False` — so `straightness` produces
**zero** edges anywhere near a dwell cluster (silently blind to it).
`local_straightness`'s `0.0` there produces a clean falling/rising pair
instead. Verified directly with a synthetic array reproducing both
signals' known values at a dwell. **Conclusion**: combining the two
signals is currently "astonishingly robust" per project owner partly
*because* `local_straightness`'s known regression happens to compensate
for `straightness`'s blind spot — a real, useful, but somewhat fragile
compensating-errors situation, not two independently-correct signals
reinforcing each other. Worth remembering if either metric's behavior
changes later.

## Cross-cutting theme: metric/data alignment ("binding contract")
Recurring failure mode this session, confirmed multiple times: a
metric that "eats" points at a boundary (difference, window, etc.) needs
an explicit, consistent convention (head-aligned / centered / tail-aligned)
relative to the raw sample it describes — silent misalignment doesn't crash,
it just quietly corrupts any downstream comparison between metrics.

**Real, concrete instances found and fixed this session**:
- `t` is **character-relative**, not stroke-relative, in the raw npy data.
  A test script forgot to normalize `t -= t[0]` per stroke before
  `searchsorted`-matching a candidate cut time, silently returning index 0
  for every stroke after the first. Caught by a debug sweep, not by
  inspection.
- ベ's real zigzag peak was off-by-one between two different scripts,
  traced to the unpadded `straightness(xy)` array (length n-2) having its
  index `i` actually correspond to raw index `i+1`.
- `forward_speed`/`backward_speed`: a copy/paste round produced a function
  named "forward" with left-padding (tail-aligned) semantics, then a
  "fixed" version left a stale docstring and a leftover `"forward_speed"`
  feature key inside `backward_speed`'s `clone()` call.
- `tangential_acceleration`: `stroke_matrix` (undefined variable),
  `stroke.featues` (typo), and an initial version that returned a bare
  `np.gradient(...)` array instead of `stroke.clone(...)`, breaking the
  `Stroke -> Stroke` compose() contract.

**Decision on how to guard against this going forward**: explicit
convention in code + comments, treated as a "binding contract," but
**not** enforced via extra machinery (e.g. an alignment tag in `props`
checked at combination time) — project owner declined that as more
formality than wanted, consistent with the "as simple as possible" ethos
and a stated lean toward **full FP, no classes** for the rewrite.

### Cut-then-smooth vs. smooth-then-cut (also an alignment question)
Tested directly: the two orderings **do** diverge in general (confirmed on
左, 文, 森 — up to ~0.0003 in normalized coords at sigma=1.0), and are only
identical when the discarded pre-cut points happen to be exact duplicates
(as in 字 stroke 0, which is why an initial quick test wrongly looked like
zero difference). Mechanism: cut-then-smooth pads the new boundary by
repeating the first kept point (`mode='nearest'`); smooth-then-cut lets the
smoothing kernel legitimately blend in real motion from the
about-to-be-discarded artifact region. **Recommendation given**: cut first,
then smooth — a synthetic "didn't move yet" stub seems a more honest
failure mode than leaking the artifact itself into the kept geometry, but
the divergence is small at sigma=1.0 for most of the checked points, so
this isn't urgent to fix.

## Edge-detection scheme (in progress, not yet finalized)
Project owner's from-scratch idea for finding the stable region (region 3)
by walking `straightness` and flagging threshold crossings.

**Simplification arc**: started with 4 edge types (full/partial ×
rising/falling), collapsed to 2 (`rising`, `falling`) with a single
threshold, plus proposed invariants (both edge lists non-empty, list
lengths differ by ≤1, alternation-implied start/end typing). Feedback
given: the invariants are corollaries of one fact (strict alternation),
not independent things to check; a single threshold will chatter without
hysteresis (need `t_re > t_fe`, Schmitt-trigger style — not yet
implemented in the real code); **10/95 real strokes have zero tail edges**,
so "both lists non-empty" is already falsified by the manual dataset and
needs an explicit empty-side case; peeling needs a hard depth cap (1 pair
per side) or it risks tunneling through a real embedded feature near the
boundary (a hane sitting close to the true edge).

**Refinement**: "start artifact mandatory, end artifact optional" — this
does fully resolve two things (edges list is guaranteed to open `rising`;
`len(fe)==0` is now a fully decidable, legitimate "no tail artifact" case,
matching the 10/95 clean-tail strokes exactly) but does **not** resolve the
core wanted-vs-unwanted ambiguity for strokes that *do* have a tail-adjacent
edge — that's the same unsolved problem from session 1 (real hane vs. noise
tail), just now localized specifically to "is the last edge, when one
exists, real content or artifact."

### Actual `detect_edges` implementation, reviewed
Vectorized the original loop-based edge detector (straightforward
`np.diff` + `np.flatnonzero` + `np.where`, verified to match the loop
exactly). Real implementation (multi-signal fusion version, accumulates
edges across repeated calls with different `signal_key`s via
`stroke.props['edges']` + `heapq.merge`) reviewed and found to have:
- **Crash risk, not yet fixed as of last check**: `edges[0][0]` /
  `edges[-1][0]` at the end has no empty-list guard. Given the effective
  threshold in use (0.2, well below the 0.293 staircase margin — see
  below), this is likely to trigger, not a rare corner case.
- **Real order-dependency bug between the two cleanup passes**,
  demonstrated concretely: `drop_same_adjacent` groups by *consecutive
  same type only*, with no time-gap check — a genuine 2-vs-1 signal
  disagreement at the same timestamp can get one of the two agreeing
  votes silently merged away into a same-type run with a much later,
  unrelated edge, before `filter_contradictions` (which groups by
  timestamp) ever sees the disagreement. The result isn't even flagged
  as a tie — the contradiction disappears without a trace. Demonstrated
  with a synthetic 4-tuple list; project owner acknowledged, said results
  are "astonishingly robust" on real data so far anyway, moving on with
  empty-list guards as the immediate fix rather than restructuring the
  two passes.
- No hysteresis in this jump-based (`abs(diff) > threshold`) approach —
  a different chatter mode (alternation) than the two-threshold Schmitt
  idea was designed to prevent; `drop_same_adjacent`'s `groupby` can't
  catch alternating chatter at all, since alternating types never form a
  same-type run.
- Edges now stored as `(type, time)` rather than `(type, index)` — enables
  merging across signals on a shared time axis, but reintroduces the same
  index↔time round-trip risk as the earlier `t`-normalization bug if
  anything downstream needs to convert a time value back to a raw index.
- **Threshold in actual use is 0.2** (not the misleading 0.9 default),
  deliberately chosen below `1.0 - 0.707 = 0.293` so the detector also
  catches partial excursions down to the known staircase-artifact value,
  not just full 0↔1 swings.

## Refactoring in progress (project owner, not yet done)
- `Character.of_strokes`: inverse of `of_npy`, reconstructs a character
  from individually-trimmed strokes. **Open question, flagged not
  resolved**: how to handle `t` once independently-trimmed strokes no
  longer respect the original character-relative, contiguous, monotonic
  contract — renormalize to close the gaps, or keep absolute timestamps
  and accept gaps.
- `Stroke.transform(fn) -> Stroke`: general transform, covering `trim`,
  `chaikin` (corner-cutting/smoothing), `rdp` (Douglas-Peucker
  simplification). **Key distinction raised**: `trim` and `rdp` preserve
  the `Stroke` contract (every surviving point has real `t`/`pressure`
  behind it); `chaikin` generates synthetic interpolated points with no
  real `t`/`pressure` to inherit, and needs an explicit policy (return a
  different, honestly-smaller type; or interpolate `t`/`pressure` too and
  accept they're now synthetic). **Resolved via project owner's move
  toward full FP** (no `Stroke` class at all) rather than a type-hierarchy
  fix — in that design `chaikin` just naturally returns a differently-
  shaped record with no `t`/`pressure` fields, no inheritance/contract
  problem to solve in the first place.
- General note on OOP drift acknowledged by project owner: `Stroke` was
  originally meant to be nothing but "the thing metrics collect on," and
  organically attracted more responsibility (God Class pattern) — part of
  the motivation for the FP pivot.

## Open items / next steps (as of end of session)
1. Add empty-list guards to the `detect_edges` family (in progress).
2. Reconcile the "straightness=0 at almost all manual cuts" claim against
   the direct check showing `1.0`/`NaN` at the exact logged ms values —
   deferred, revisit now that full logging is done.
3. Fix or explicitly accept `local_straightness`'s dwell→`0.0` regression,
   given it's now known to be doing real (if accidental) work in the
   edge-detection combination.
4. Decide `Character.of_strokes`' `t`-reconstruction policy.
5. Finalize `chaikin`'s return type/contract under the FP redesign.
6. Build the actual cut-point × metrics matrix (parked mid-session in
   favor of the edge-detection idea; still the natural next full deliverable).
7. Add hysteresis (two thresholds) or a time-gap-aware version of
   `drop_same_adjacent` to `detect_edges` if the order-dependency bug
   ever actually manifests on real data (currently latent, not observed).
8. Fixed vs. arc-length-scaled window sizing for regions 1/5 — parked,
   explicitly to be decided from real numbers once the matrix exists, not
   from the physiological literature alone.
9. FSM formalization of the 5-region model — early-stage idea, matrix
   should come first as calibration data; watch for dwell-cluster
   chatter requiring debounce/hysteresis in any transition rule.