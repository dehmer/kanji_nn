# Handoff: Kanji/Kana Stroke Cut Detection

## Project
Cleaning up Kanji/Kana stroke vector data captured via Flutter app on iPad/Apple
Pencil. Known artefacts at pen-down/pen-up (staircase quantization, micro-jitter,
sensor sentinels). Goal: find exact head/tail trim indices per stroke, using
metrics-derived edge detection - no a-priori smoothing/resampling (noise seems
to help, not hurt, the clean-up process). This is an exploratory pet project,
explicitly not "good enough software" - simplicity favored over completeness.

## Data model (established, in code)
- `raw` per stroke: `[t, x, y, pressure]`, columns fixed in that order.
- Multi-stroke `Character.raw` has a 5th pen-state column (1...1,0 per stroke);
  `split_strokes`/`join_strokes` round-trip on it.
- `Stroke` is an immutable `@dataclass(frozen=True, kw_only=True)` (kw_only
  added this session to stop fragile positional construction as fields grew).
  Carries `raw`, `features` (per-sample arrays), `props` (per-stroke scalars/
  tuples, e.g. `props["cuts"] = (head_cut, tail_cut)`), plus dataset/code_point/
  literal/stroke_type identity fields. `.trim(region)` slices raw and wipes
  features/props (trimming happens *after* scoring a prediction, not inside
  the metrics pipeline).
- Metrics live one-per-file in a package, chained via a `compose(...)` helper
  that runs the list **bottom-to-top**. Each metric function takes a `Stroke`,
  returns `stroke.clone(features={...})`. Available: `arc_length` (s, s_norm,
  ds), `central_speed`, `forward_speed`, `backward_speed`, `tangent`,
  `curvature`, `straightness` (misleadingly named - it's actually ~always 1,
  a tangent-unit-length sanity check per its own docstring, not a speed
  measure), `local_straightness` (3-point chord/arc ratio, genuinely useful),
  `pressure_derivative` (dP/dt), `vector_acc`, `tangential_acc`. A custom
  `pressure()` step copies raw pressure into features and adds `pressure_norm`
  (per-stroke min-max normalized; min is always 0 in practice, see sentinels
  below).
- `do_the_magic_trick(stroke)` is the pipeline's cut-detection stub/hook -
  reads `stroke.stroke_type` (3-tuple: literal, code_point, name e.g.
  `"CJK STROKE H"`), dispatches to a `detectors` dict keyed by type name, falls
  back to a no-op `default_detector` for unregistered types.
- CSV ground truth (`analysis-short-samples.csv`): columns
  `dataset, literal, code_point, stroke_idx, head_cut, tail_cut`. **head_cut
  inclusive, tail_cut exclusive** (`raw[head_cut:tail_cut]`, matches
  `Stroke.trim`). `tail_cut == n_points` means "no tail cut" (falls out
  naturally, not a special case). Cuts were eyeballed by the user, single-
  stroke-at-a-time (no full-character context) - the ones with no tail_cut
  are genuine "couldn't decide," not confident negatives. Treat CSV as
  directional/rough calibration, not strict ground truth, especially on
  ambiguous rows. Convention should stay the same for any new CSV rows.

## Hard-won data facts
- **Pressure sentinels, exact and deterministic across every stroke in the
  95-stroke/13-character sample checked:** `pressure[0] == 0.08` always;
  `pressure[-1] == 0.0` always. Index 0 and index n-1 are *guaranteed*
  contaminated by definition - not something to detect, something to assume.
  `pressure[1]` is 0.08 in ~94% of strokes (sentinel sometimes holds one extra
  sample); `pressure[-2]==0` in only ~15% (tail sentinel is basically just the
  single last sample, not a run).
- **Staircase artefact is directly visible in raw x/y** (literal repeated
  coordinate values for several consecutive samples, then a step) - not just
  an inference from noisy derivatives. Shows up in slow-moving sections
  generally, not only at stroke boundaries.
- **`ds` (from arc_length, un-epsilon'd, consecutive-point distance) is
  jitter-immune** - no dt division, so it's unaffected by Flutter/iPad
  timestamp irregularities (dt sometimes 3-14ms against an ~8ms median).
  Good raw local signal for the staircase dwell.
- **Corner (kado) vs. hook-flick (hane) vs. real lift artefact all look
  similar in position-derived metrics but differ in pressure:**
  - A sharp direction-change corner barely disturbs `local_straightness`
    (traversed smoothly in one direction) but does put a mild trough in
    speed/ds.
  - A hane (reversal/hook) causes `local_straightness` to dive hard (~0),
    but pressure stays essentially flat/unaffected - pressure not reacting
    is what tells a hane apart from a real lift.
  - A genuine artefact wobble at a tail also dips `local_straightness` hard
    AND coincides with real pressure decline - the combination is the tell,
    neither channel alone is reliable (confirmed on concrete examples: 字
    stroke 2 vs 貝 stroke 1).
- **Accelerating vs. decelerating tail is a real, load-bearing discriminator
  for "genuine hane at the very end" vs "artefact tail":** 字 stroke 4's
  ground truth has *no* tail cut despite `central_speed` peaking at its
  highest value in the whole stroke in the last few samples (a genuine
  calligraphic flick into the lift). Contrast 貝 stroke 1, where the tail
  *is* cut, and speed is decelerating into the lift there. Confirmed this
  aligns with `kvg:type` being a hook type (`WG`) for 字/4 - see below.
- **Sigma-lognormal (Plamondon) is the wrong reference shape for sustained
  strokes.** It models a single ballistic point-to-point movement (skewed,
  no cruise phase). What's actually seen in a longer/straighter stroke (e.g.
  左/0) is closer to a **trapezoidal velocity profile**: roughly symmetric
  rise/plateau/decline in thirds, platykurtic. Use trapezoidal-shape
  expectations for sustained strokes; lognormal reasoning may still apply to
  short ballistic flicks specifically, untested.
- **Short strokes (e.g. flicks) can have ~50% of total duration classified as
  head+tail artefact combined**, split roughly evenly - a real stress case for
  any "find the stable interior region" approach, since there's very little
  clean baseline to calibrate against.

## KanjiVG `kvg:type` integration (new, working)
- Unicode CJK Strokes block (U+31C0-U+31EF). Base codepoint name is reliable
  (get via Python stdlib `unicodedata.name(chr(...))`, no dependency needed,
  confirmed exact match against 3 independent sources). Letter suffixes
  (`a`, `b`, `va` etc.) are **undocumented even by the KanjiVG maintainer**
  (open issue) - don't rely on them.
- Hook-family types (name ends in `G` = gou/hook): `HG` (horizontal-hook,
  U+31D6), `SG` (vertical-hook, U+31DA), `WG` (bent-hook, U+31C1), `SWG`
  (vertical-bend-hook, U+31DF). `is_hook(type_cp)` can be derived from the
  name suffix rather than hardcoding a list.
- `N` (right-falling/"press", U+31CF, 捺/na) is a third distinct ending
  pattern worth tracking separately later: physically a deliberate press/
  widen motion, not a lift-decay or an accelerating flick. Not yet
  investigated in the kinematic data (only flagged as a hypothesis).
- Across the 13-character/95-stroke sample: 10 distinct types present (of 29
  possible). Counts: H=29, S=18, P=16, D=14, N=5, HG=5, HZ=3, WG=2, SG=2,
  SWG=1. **Hook types combined = 10/95 instances** - real but thin per-type;
  decided to treat hook-vs-not as one binary rather than 4 separate detectors
  for now, given how few instances some individual hook types have.
- KanjiVG does **not** cover Hiragana/Katakana (confirmed) - kvg:type is
  explicitly scoped as "training wheels for kanji, to be ditched once the
  underlying kinematic signal is understood," not a permanent dependency.
- Stroke-order alignment between KanjiVG and the captured data is guaranteed
  by construction: the Flutter app validates each written stroke against the
  KanjiVG template and only accepts a character if stroke count matches - no
  possibility of misalignment for this dataset.
- DB: Postgres table `kvg_type(literal, stroke_idx, type_literal, type_cp)`,
  keyed by `literal` (not code_point - watch for literal-text collisions
  across distinct code points if ever extending beyond kanji; not an issue
  currently since kanji variants were explicitly excluded from extraction).
  Python access: `psycopg` (v3, not v2 - v3 is the actively-developed
  successor, v2 is maintained but feature-frozen), pool-less lazy
  module-level singleton connection (justified: strictly synchronous,
  single client).

## Detection approach - what was tried, what to avoid repeating
An earlier (separate, discarded) solo attempt exists in the codebase as dead
imports (`detect_edges`, `consolidate_edges`) - ignore, not a starting point.

This session's iteration (single global composite signal, symmetric
head/tail treatment) went through several rounds and is a **cautionary
tale, not a foundation to build on**:
1. `composite = sqrt(v_norm * pressure_norm)` (soft-AND), single percentile
   threshold from the stroke's own middle-band, symmetric scan from both
   ends - reasonable first pass, but a **single percentile threshold is
   fragile whenever the "stable middle band" itself contains a genuine
   mid-stroke stall** (kado/hane), which drags a low percentile down and
   causes false early triggers on head jitter.
2. Switching to median-based threshold fixed head cases but broke tail
   cases badly - **head and tail have genuinely different "clean" dynamics**
   (tail's legitimate clean region can have lower activity than head's, since
   natural deceleration into a stop is normal there). Confirms head/tail
   need separately-tuned logic, not one threshold serving both.
3. Split to `composite`-based head detection + `pressure_norm`-alone tail
   detection - fixed the regression, but lost the very discriminator (speed
   direction) needed to keep the accelerating-hane case (字/4) uncut - it got
   overcut.
4. Added a speed-trend gate (fit slope of `central_speed` in a window before
   the tail candidate; positive slope -> override to no-cut) - fixed 字/4
   exactly, but **over-fired on other strokes** (false "no cut" on real
   cut cases) because a short-window slope-sign test on one channel is too
   sensitive to noise.
5. Tried requiring corroboration from `ds` alongside `central_speed`
   (both must show positive slope) - **broke the original target case again**
   (字/4's true signal was concentrated in the boundary sample the window
   deliberately excludes; `ds` disagreed with `central_speed` in the
   remaining window) while only partially fixing the false positives
   elsewhere (both channels can be correlated/noisy together, so an AND
   didn't add much real independence).

**Lesson explicitly agreed with the user: stacking one-off patches against
individual failing rows produces exactly the kind of bespoke complexity the
project is trying to avoid.** User called a stop here and pivoted to a
different strategy (below) rather than continuing to patch.

## Current strategy (in progress, this is the live approach)
- **No more single global detector.** Head and tail are detected with
  **completely separate metric sets/logic** chosen per-side as needed - no
  shared threshold, no assumption of symmetry, additional metrics allowed
  if useful.
- **Dispatch by `kvg:type` name.** `detectors = {"CJK STROKE H": ..., ...}`
  dict, `do_the_magic_trick` looks up `stroke.stroke_type` name, falls back
  to a no-op default. One detector built and validated per stroke type,
  starting with the most common type first (not all 29 at once - grounded in
  actual per-type instance counts, decided against the earlier temptation to
  plan for all types before having examples of most of them).
- **Method for building each detector:** for a specific stroke, describe the
  multi-channel plot in plain-English/structured docstring form (co-located
  with the detector code, not separate documentation), identify **rising/
  falling edge candidates per channel with an explicit notation**:
  ```
  Legend: abc(i/j[f]) = rising edge on abc between i,j. abc(i\\j[f]) = falling edge.
          f = '*' both endpoints viable, f = <idx> only that endpoint viable.
  Boundaries/indices strictly inclusive: [a, b].
  ```
  Then check where multiple independent-looking candidates cluster - a tight
  cluster across several channels is treated as one real event viewed through
  multiple lenses (not necessarily N independent votes), and a simple
  "which index is shared across the most candidate edges" rule is the
  current consolidation approach (no dedicated picker function yet, see
  below - not needed until a genuinely ambiguous multi-candidate case shows
  up).
- **First concrete result under this approach, validated against ground
  truth:** 左 (U+5DE6) stroke 0, type `H` (plain horizontal, most common type
  in the dataset, 29/95 instances). Ground truth `head_cut=17`. Four
  candidate signals (`ds` rising edge 17/18, `local_straightness` rising edge
  16/17, `pressure_norm` window 15-18, `dP/dt>0` over 15-18) all cluster
  tightly; index 17 is shared by the two edge-based candidates and sits
  inside the others' windows. Simple "most-shared index" pick lands exactly
  on ground truth, no tie-break logic needed. **Tail detection for this
  stroke not yet started.**
- **`pick_candidate` design decision (not yet built):** should be injected
  via `partial` at composition time (`partial(do_the_magic_trick,
  pick_candidate=...)`), matching the existing pattern already used for
  `tangential_acc`'s `speed_key` - not threaded as a constructor parameter
  through `Character`/`Stroke`/pipeline. Deliberately deferred - build only
  once there's a real case with genuinely non-trivial disagreement between
  candidates to choose from, per user's explicit "less is more" preference.
  **User is currently working on this alone by request - do not prompt
  further on this specific piece unless asked.**

## Immediate next steps (as of handoff)
1. User is prototyping something to produce one (or all four) of the 左/0
   head candidates programmatically - solo, no input requested yet.
2. After that: compare notes, then move to **tail detection for 左/0**
   (type `H`).
3. Then generalize the `H` detector to the other 28 `H`-type instances in the
   dataset, adjust for problems found, and only then update the CSV row(s)
   if satisfied - CSV updates are provisional/incremental, tied to specific
   validated detector coverage, not a one-shot bulk edit.
4. Eventually repeat the same per-type build/validate loop for `S`, `P`, `D`,
   then the thinner hook-family types (treated as one `is_hook` binary
   rather than 4 separate detectors, and `N`/press-ending strokes as a third,
   separately-flagged ending pattern, not yet investigated kinematically).