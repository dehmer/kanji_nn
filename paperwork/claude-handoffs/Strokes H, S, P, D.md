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
  stops fragile positional construction as fields grew - e.g. `code_point`
  and `literal` are both strings and easy to swap by position). Carries
  `raw`, `features` (per-sample arrays), `props` (per-stroke scalars/tuples,
  e.g. `props["cuts"] = (head_cut, tail_cut)`), plus `dataset`, `code_point`,
  `literal`, `stroke_type` (3-tuple: literal, code_point, name e.g.
  `"CJK STROKE H"`) identity fields. `.trim(region)` slices raw and wipes
  features/props (trimming happens *after* scoring a prediction, not inside
  the metrics pipeline).
- `Character.stroke_types()` lazily fetches `kvg_type` rows from Postgres
  (see DB section below) and caches on the instance; `Character.strokes()`
  asserts `len(strokes) == len(types)` before zipping them into `Stroke`
  objects - deliberately kanji-only, not extended to handle kana's absence
  of type data (no kana in scope for this project; do not "fix" this
  assertion to be kana-tolerant unless the project scope actually changes).
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
- `find_trim_region(stroke)` is the pipeline's cut-detection entry point -
  reads `stroke.stroke_type` name, dispatches to a `detectors` dict, falls
  back to a no-op `default_detector` for unregistered types. Guards against
  overwriting a preset `cuts` value already on the stroke.
- CSV ground truth (`analysis-short-samples.csv`): columns
  `dataset, literal, code_point, stroke_idx, head_cut, tail_cut`. **head_cut
  inclusive, tail_cut exclusive** (`raw[head_cut:tail_cut]`, matches
  `Stroke.trim`). `tail_cut == n_points` means "no tail cut" (falls out
  naturally, not a special case). Cuts were eyeballed by the user, single-
  stroke-at-a-time (no full-character context) - the ones with no tail_cut
  are genuine "couldn't decide," not confident negatives. Treat CSV as
  directional/rough calibration, not strict ground truth, especially on
  ambiguous rows. CSV updates are meant to be incremental, tied to specific
  validated detector coverage, not a one-shot bulk edit.

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
  This turned out to be enough on its own for the "simple shape" detector
  (see below) - `pressure_norm`/`central_speed` were useful for building
  intuition but the shipped detector needs only `ds`.
- **Corner (ore/折れ) vs. hook-flick (hane/跳ね) vs. real lift artefact all
  look similar in position-derived metrics but differ in pressure:**
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
  aligns with `kvg:type` being a hook type (`WG`) for 字/4 - see below. This
  is exactly the kind of case the current shipped detector (simple-shape
  strokes only) is *not* meant to handle - see "current state" below.
- **Sigma-lognormal (Plamondon) is the wrong reference shape for sustained
  strokes.** It models a single ballistic point-to-point movement (skewed,
  no cruise phase). What's actually seen in a longer/straighter stroke is
  closer to a **trapezoidal velocity profile**: roughly symmetric
  rise/plateau/decline in thirds, platykurtic. Use trapezoidal-shape
  expectations for sustained strokes; lognormal reasoning may still apply to
  short ballistic flicks specifically, untested.
- **Short strokes (e.g. flicks) can have ~50% of total duration classified as
  head+tail artefact combined**, split roughly evenly - a real stress case for
  any "find the stable interior region" approach, since there's very little
  clean baseline to calibrate against.

## KanjiVG `kvg:type` integration (working)
- Unicode CJK Strokes block (U+31C0-U+31EF). Base codepoint name is reliable
  (get via Python stdlib `unicodedata.name(chr(...))`, no dependency needed,
  confirmed exact match against 3 independent sources). Letter suffixes
  (`a`, `b`, `va` etc.) are **undocumented even by the KanjiVG maintainer**
  (open issue) - don't rely on them.
- Hook-family types (name ends in `G` = gou/hook): `HG` (horizontal-hook,
  U+31D6), `SG` (vertical-hook, U+31DA), `WG` (bent-hook, U+31C1), `SWG`
  (vertical-bend-hook, U+31DF). Can derive "is hook" from the name suffix
  rather than hardcoding a list.
- `N` (right-falling/"press", U+31CF, 捺/na) is a third distinct ending
  pattern worth tracking separately: physically a deliberate press/widen
  motion, not a lift-decay or an accelerating flick. Not yet investigated in
  the kinematic data. User has personal Shodo (brush calligraphy) experience
  with this stroke and flagged it as exceptionally hard to write even by
  hand - noted as a possible (untested, not to be overstated) source of
  motor-habit carryover into stylus writing, worth revisiting only if N
  strokes show a distinctive signature once actually examined.
- Across the 13-character/95-stroke sample: 10 distinct types present (of 29
  possible). Counts: H=29, S=18, P=16, D=14, N=5, HG=5, HZ=3, WG=2, SG=2,
  SWG=1. Hook types combined = 10/95 instances - real but thin per-type,
  treated as one binary rather than 4 separate detectors if/when needed.
- KanjiVG does **not** cover Hiragana/Katakana (confirmed) - kvg:type is
  explicitly scoped as "training wheels for kanji, to be ditched once the
  underlying kinematic signal is understood," not a permanent dependency.
- Stroke-order alignment between KanjiVG and the captured data is guaranteed
  by construction: the Flutter app validates each written stroke against the
  KanjiVG template and only accepts a character if stroke count matches - no
  possibility of misalignment for this dataset.
- DB: Postgres table `kvg_type(literal, stroke_idx, type_literal, type_cp)`,
  keyed by `literal` (not code_point - watch for literal-text collisions
  across distinct code points if ever extending beyond kanji; kanji variant
  entries were already found duplicated once during extraction and have
  been excluded). Python access: `psycopg` (v3, not v2 - v3 is the actively-
  developed successor, v2 is maintained but feature-frozen), pool-less lazy
  module-level singleton connection (justified: strictly synchronous,
  single client).

## Current state: `find_trim_region.py`
A single detector, `CJK_STROKE_H`, handles strokes with a **simple shape**
(no hane/ore) by finding the peak of `ds` and searching outward from it in
both directions for the first "not moving" (`ds == 0`) sample:

```python
def CJK_STROKE_H(stroke):
    ds = stroke.features["ds"]
    ds_max_idx = np.argmax(ds)
    mask = (ds > 0)
    find_index = lambda rng: next((i for i in rng if not mask[i]), rng[-1]) if len(rng) else 0
    ranges = [
        range(ds_max_idx - 1, -1, -1),      # backward search, left slope
        range(ds_max_idx, stroke.n_points)  # forward search, right slope
    ]
    cuts = tuple(find_index(r) for r in ranges)
    return stroke.clone(props={"cuts": (cuts[0], cuts[1] + 1)})
```

- **Validated (battle-tested) against ground truth for CJK STROKE H, S, P,
  and D** - the "simple shape" types, no hook/hane/ore expected. Reliable
  and robust; in the user's own assessment, better than eyeballing.
- **Deliberately does not and should not be used for strokes with hane or
  ore** - it has no concept of "this speed/direction change is a real
  calligraphic feature, not artefact." This is by design, not a gap to
  patch onto this function - it needs a genuinely different detector, not a
  parameter tweak to this one (see "lessons learned" below for why patching
  in place is exactly the trap to avoid repeating).
- **`detectors` dict currently also maps HG, WG, N, HZ, SG, SWG to
  `CJK_STROKE_H`** with a code comment noting it's known *not* to work well
  for them ("P_norm still high in ds troves"). **This is a known, called-out
  placeholder, not a considered decision** - worth either building real
  detectors for these types next, or mapping them to `default_detector`
  explicitly in the meantime so a wrong-but-plausible-looking cut doesn't
  silently ship for a stroke type the user already knows this detector
  handles badly.
- **Fixed bug, worth knowing about if this function gets copied/modified
  further:** the backward search range (`range(ds_max_idx - 1, -1, -1)`) is
  *empty* whenever `ds_max_idx == 0` (the `ds` peak sits at the very first
  sample) - about 11% of random trials in stress-testing, i.e. not a rare
  edge case. The original `find_index` lambda's fallback (`rng[-1]`) crashes
  with `IndexError` on an empty range. Fixed by adding `if len(rng) else 0`
  before the fallback. **Verified this is asymmetric**: the forward/tail
  range can never be empty (it always contains at least `ds_max_idx` itself,
  since that's a valid index by construction), so no equivalent guard is
  needed on that side - checked explicitly with 20,000 forced-peak-at-last-
  index trials, zero crashes. If extending or rewriting this function,
  remember "non-degenerate" for `ds_max_idx` has more than one distinct
  failure shape (empty array entirely vs. peak landing exactly at an edge)
  and each needs its own guard/assertion, not one blanket check.

## Lessons learned this session (read before extending detectors further)
- **An earlier (separate, discarded) solo attempt exists in the codebase as
  dead imports** (`detect_edges`, `consolidate_edges`) - ignore, not a
  starting point.
- **A single global composite-signal detector (symmetric head/tail
  treatment) was tried and explicitly abandoned mid-session** after several
  rounds of patch-on-patch (percentile threshold too sensitive to mid-stroke
  stalls -> switched to median -> broke tail -> split head/tail signals ->
  lost the accel/decel discriminator -> added a speed-trend gate -> gate
  over-fired -> tried requiring `ds` corroboration -> broke the original
  target case again). **The recurring failure mode was optimizing against
  individual failing rows instead of building from a principled hypothesis
  - each fix was locally reasonable but the accumulation was exactly the
  kind of bespoke complexity this project is trying to avoid.** The
  `kvg:type`-dispatched, one-type-at-a-time approach that replaced it (and
  produced the validated H/S/P/D detector above) worked specifically
  because it stayed narrow and validated each piece before moving on -
  worth deliberately protecting that discipline when extending to
  hane/ore types, rather than reaching for one clever detector meant to
  cover everything at once.
- **Verify index arithmetic numerically, every time, rather than trusting
  verbal/algebraic re-derivation** - even late in this session, multiple
  attempts to re-express the same verified-correct logic in a cleaner style
  (functional-programming refactor, config-driven refactor) introduced real
  off-by-one bugs that looked plausible on inspection and were only caught
  by exhaustive randomized stress-testing against the original, trusted
  implementation. This applies doubly when "cleaning up" or "refactoring"
  code that already works - matching behavior exactly is the actual bar,
  not "looks equivalent."
- Also worth remembering: `arr[:]` on a numpy array is a genuine no-op
  (confirmed), but `some_slice_object[:]` is not valid at all (`slice`
  objects aren't subscriptable) - easy to conflate the two.

## Immediate next steps
1. Resolve the HG/WG/N/HZ/SG/SWG placeholder mappings - either build real
   detectors for these hane/ore-containing types, or explicitly route them
   to `default_detector` in the meantime rather than leaving them silently
   mapped to a detector already known to misbehave for them.
2. Design detector(s) for hane/ore-containing stroke types. This is
   genuinely a different problem from the simple-shape case - re-read the
   "hard-won data facts" above (kado vs. hane vs. artefact, accel/decel
   tail discriminator, pressure lag) before starting, since this ground was
   covered once already this session via a since-abandoned approach and the
   findings there are still the best available leads.
3. Continue the per-type build/validate/update-CSV loop established this
   session: pick a type with a decent instance count, build and check
   against ground truth, update CSV only for validated coverage, move to
   the next type.
4. `pick_candidate` (a consolidation strategy for multiple near-equivalent
   candidate cut indices, injected via `partial` at composition time rather
   than threaded through `Character`/`Stroke`) was discussed and
   deliberately deferred - only build it once a detector produces a
   genuinely ambiguous multi-candidate case to resolve, not preemptively.