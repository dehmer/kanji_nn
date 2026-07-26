# Handoff: HZ Tail-Cut Detection, CPD Infrastructure, Curvature Fix

## Context
Follow-on session to the original "Strokes H, S, P, D" handoff. Scope this
session: build working head/tail cut detection for the "bimodal" (hane/ore)
stroke types, starting with the reference type CJK STROKE HG, then pivoting
to CJK STROKE HZ as the higher-value target once the population grew.
Population grew from 13 characters/95 strokes to 80 characters/400 strokes
(`kanken-10_80`, Kanken Level 10). Same low-quality-bar, exploratory,
rapid-iteration philosophy as before; heavy (un-)commenting workflow.

## Population / stroke-type distribution (kanken-10_80, 400 strokes)
H=136, S=88, P=61, D=33, HZ=22, N=18, HG=6, SG=6, HZG=6, SWG=5, SW=4, WG=4,
HP=3, T=2, HZWG=2, PZ=2, SP=1, PD=1.

HZ (22 instances: 3 already embedded in 貝/足/音 from the original 18, plus
19 freshly annotated: 車四五早目虫田見草男口町中百白石右日名) became the
active target type instead of HG, purely due to instance count.

**Naming insight (new):** CJK stroke-name letters are a documented,
principled decomposition (H=héng horizontal, S=shù vertical, P=piě
left-falling, N=nà right-falling, T=tí rising, W=wān curved, X=xié slanted,
Z=zhé bent/corner, G=gōu hook, Q=quān circle — source: BabelStone CJK
stroke glossary). Compound names spell out their component primitives
literally (e.g. HZZZG = horizontal + 3 corners + hook, confirmed real,
U+31E1, appears 14/6355 times in Kanken). This *updates* (does not
contradict) the original handoff's caution: that caution was specifically
about KanjiVG's own `kvg_type` *variant suffix letters* (`a`, `b`, `va`),
which remain genuinely undocumented — the Unicode stroke-name letters
themselves are a different, reliable thing.

## `CJK_STROKE_H`: found to have a general vulnerability, not just a
bimodal-types problem
Confirmed on 文 stroke 3 (type N, previously assumed unimodal-safe):
`CJK_STROKE_H`'s tail search stops at the *first* `ds==0` sample. If a
staircase dwell happens to touch exact zero mid-stroke (before the real
tail), the cut lands there instead of the true end — even though `c_speed`
never fully collapses and real motion resumes for a long stretch
afterward. This means the type-based "H/S/P/D safe, hane/ore unsafe" split
is not as clean as assumed; it's really about whether *any* mid-stroke
pause touches zero, which can happen for any type. N was provisionally
"demoted" to unimodal-but-not-currently-fit-for-duty as a result.

## Hane/ore tail-cut detector: iteration history (all on CJK_STROKE_HG,
then generalized)

1. **Global running-minimum trough-finder** — failed badly (errors up to
   ±33). Root cause: locks onto the *first* exact-zero sample anywhere in
   the tail (often a mid-stroke corner/pause), not the true final decay,
   because it tracks lowest-value-ever rather than the relevant local
   minimum.
2. **Last local minimum (falling→rising transition)** — fixed the above
   (mean |error| ~5.1), but missed cases with a terminal decline that
   never turns back upward before the recording ends (e.g. 学-4).
3. **+ terminal-decline override** (if still falling at array end and it's
   a new global low, treat as the real final trough) — fixed 学-4 exactly,
   but over-corrected on strokes with small-amplitude noise wobble near
   the very end (e.g. 足-6, error went from -4 to -10): the override
   can't distinguish a genuine final decay from low-amplitude pen-up
   noise, since it only looks at direction/global-min, not amplitude
   relative to the stroke's own motion scale.
4. **Reframed as change-point detection (CPD)** rather than trough-hunting
   — see below. This is the line of work still active.

## CPD (change-point detection) work

### Core method: mean-difference sweep
Brute-force scan: for each candidate split `t`, compare
`mean(S[t-w:t])` vs `mean(S[t:t+w])`; the `t` maximizing the absolute
difference is the predicted change point. Pure NumPy, translates cleanly
to Dart. Files: `find_change_point.py`, `find_change_point_adaptive.py`,
`mean_diff_sweep.py` (near-duplicates from iterative experimentation —
worth collapsing to one canonical version later).

### Combined signal
`cpd_signal.py` / `compute_combined_signal.py`: z-score-normalize 2-3
feature channels, combine via weighted sum, feed the result into the
sweep. Channels transformed so all "increase toward the dirty tail zone"
(e.g. `P_inv = 1 - P_norm`).

### Bugs found and fixed this session
- **Window-size vs. search-fraction interaction**: fixed `window_size`
  (e.g. 8) combined with a percentage-based `start_search_idx` (e.g. last
  15%) can leave almost no valid candidate range on shorter strokes —
  confirmed concretely on 石 (`n=87`): search range only covered 6
  candidate indices, and the true answer (83) was structurally outside
  the window entirely. Not a signal-quality problem — the sweep could
  never have found the right answer regardless of channel weights.
  **Fixed** by switching to `find_change_point_adaptive`-style window
  sizing (`window_pct`, `min_w`, `max_w` instead of a fixed sample count).
  One-variable-at-a-time discipline: with `window_pct=0.06` alone (nothing
  else changed), 石's error went from -9 to -2, and the *whole* HZ set's
  mean error dropped from -2.95 to -0.5 (MAE 3.6 → 2.5), with the error
  distribution flipping from systematically one-directional to roughly
  symmetric noise (±5 max, no outliers). Current known-good config:
  `fraction=0.85`, `window_pct=0.06`.
- **Degenerate-empty-search-range edge case** (flagged, not yet hit in
  practice): if a stroke is short enough that
  `start_search_idx >= n - window_size`, the sweep loop never executes and
  `scores` stays all-zero; `argmax` then silently returns index 0 rather
  than failing. Worth an explicit guard before running unattended over
  the full 80-character population.

### Weight optimization: overfitting concern, resolved in favor of
reasoning over fitting
`optimize_weights`/`optimize_pipeline` (grid search over channel weights
+ window params, minimizing MAE against the 22 HZ labels) is real
infrastructure but was flagged as high overfitting risk: searching a
multi-dimensional space against only 22 noisy-labeled examples, with no
held-out split, evaluated against the same data it's tuned on. Confirmed
this concern empirically: a 3-channel grid search (`P_inv`, `loc_stness`,
`K`) collapsed to `loc_stness: 0.95` with `MAE=1.55` — a suspiciously
clean single-channel dominance from a small, noisy-labeled search.
**Decision made:** hand-tune weights from domain reasoning (pressure
decay is the most physically guaranteed signal; `c_speed` decay and `K`
support it), and use the 22 labeled strokes as a **validation set**, not
a fitting target. First hand-tuned attempt
(`["P_inv","c_speed","K"]`, weights `[0.5,0.35,0.15]`) already
outperformed prior CPD-only attempts on known-hard cases (貝-1: -9→-5,
足-1: -8→+1).

### Full 22-stroke HZ tail-cut result (current best, hand-tuned weights +
adaptive window fix)
Mean error ≈ -0.5, MAE ≈ 2.5, max |error| = 5 (four strokes tied at
exactly ±5: 口, 白, 目, 草). No systematic directional bias remaining —
error looks like noise around the right answer, not a shared fixable
mechanism. Reasonable stopping point per project's "simplicity over
completeness" bar, or a jumping-off point for one more targeted trace if
picked back up.

### Important known limitation of the population itself
The original 18-stroke (now 22, HZ-specific) sample is **not a random
sample** — it is, by construction, the shortest strokes in the population
(by duration, point count, and arc length), i.e. the worst-case /
least-stable-baseline strokes. All tuning and validation so far has been
against this adversarial subset. Behavior on longer, more typical strokes
of the same type is not yet separately verified.

## Curvature chain (`θ`, `dθ/ds`, `K`): real bug found and fixed
Root cause chain, fully diagnosed:
1. At a staircase dwell (exact-duplicate consecutive raw x/y), the true
   tangent direction is undefined. `atan2(0,0)` silently returns `0.0` —
   a defined-but-meaningless angle, not an error.
2. Differentiating that discontinuity (`θ` jumping to/from 0) against `s`,
   which only advances by the `1e-12` anti-duplicate epsilon during the
   same dwell, produces enormous spurious values (`dθ/ds`, `K` up to
   `~10^11`–`10^12`) — confirmed on real data (貝-1).
3. This is the *same root defect* as `stness` collapsing to exactly 0
   during dwell (diagnosed earlier) — magnitude-based quantities degrade
   gracefully (collapse to 0), direction-based quantities blow up
   (undefined angle differentiated against ~0 step).

**Fix implemented:** compute `θ`/`dθ/ds`/`K` from a *separately smoothed*
copy of x/y (Gaussian, `sigma≈1`, via `scipy.ndimage.gaussian_filter1d`),
while metrics that already work well on raw data (`ds`, `local_straightness`,
etc.) keep using raw x/y — both forms coexist as separate feature channels
(`xy_smooth`, `ds_smooth`, `s_smooth`, `s_smooth_norm` added to
`arc_length.py`; `tangent.py`/`curvature.py` consume the smoothed chain).
`tangent.py` also normalizes `tx,ty` to unit length (`+1e-8` guard) —
confirmed this is doing real corrective work, not just cosmetic: since
`np.gradient` uses the same local spacing for both components, a
too-small `s` step inflates both `tx_raw`/`ty_raw` by the same factor,
so the *ratio* (true direction) survives even when magnitude is huge;
normalizing recovers the correct unit tangent.

**Two bugs found and fixed in the smoothing implementation itself:**
- `s_smooth` initially lacked the anti-duplicate epsilon that raw `s` has.
  Assumption "smoothing removes all duplicates" is false for sufficiently
  long dwell runs (confirmed empirically: 14/95 strokes still produced an
  exact-zero `ds_smooth` even after `sigma=1.0` smoothing; longest raw
  dwell run observed is 11 samples). This caused real
  `RuntimeWarning: divide by zero` + `NaN` in `tangent.py`, confirmed on
  学 stroke 4. **Fixed** by adding the same `s_smooth += arange(n)*1e-12`
  epsilon trick already used for raw `s`.
- (Noted, not yet acted on) `gaussian_filter1d` reintroduces a `scipy`
  dependency, which cuts against the "everything beyond NumPy is taboo"
  constraint stated for Dart-portability. Treated as acceptable for now
  ("common enough to reimplement later if needed") — same treatment
  extended to `kneed`.

**Validated on real data (貝-1):** after the fix, `K`/`dθ/ds` are bounded
and physically sensible (~±1000-1500, not `10^12`), with one clean sharp
spike exactly at the stroke's one real corner — matching the `HZ` type
name's decomposition (one horizontal segment, one bend) and giving a
first concrete (n=1) data point in favor of the "derive segment/corner
count from curvature instead of the DB label" idea (see Open Ideas).

## `pressure_knee.py` (new module, mirrors the metrics one-function-per-file
convention)
Uses Kneedle algorithm (`kneed` library) to find where pressure
transitions from ramp to plateau (head) and plateau to decay (tail).
**Explicitly not intended as a cut candidate** — validated for a
different purpose: bounding/narrowing a CPD search window.

- **Split point:** `ds_max_idx` (same point `CJK_STROKE_H` uses), *now
  clamped* via a `min_window` parameter so neither the head nor tail side
  of the split can drop below a minimum sample count. This clamp was
  added after stress-testing found 6/100 strokes had a tail window under
  8 samples when using the raw, unclamped `ds_max_idx` split — one case
  (学 stroke 6, n=83, ds_max_idx=80) left only 3 samples for Kneedle to
  work with. Same failure shape as CPD's window-size squeeze on 石, just
  relocated to the split point. **This clamped version has not yet been
  re-validated against the original 4-stroke tail_knee check — do that
  first before trusting it further.**
- **Validated finding (n=4: 貝-1, 足-1, 音-6, 石-3), unclamped version:**
  `tail_knee` reliably falls **at or before** the true `tail_cut` (4/4) —
  i.e. `[tail_knee, n_points)` is a search window guaranteed (so far) not
  to exclude the real answer. `head_knee` has **no such guarantee**
  (2/4 landed past the true `head_cut`, which would silently exclude it
  from a `[0, head_knee]` window) — explained by pressure's own rise
  being short/sharp (per the original sentinel facts) and often
  plateauing before the *trajectory* has actually settled.
  **Practical implication:** `tail_knee` is usable to replace CPD's
  fraction-based `start_search_idx` (would directly fix 石-style
  failures); `head_knee` is not trustworthy for the same purpose without
  further validation or pairing with a motion-based channel.
- `pip install kneed` is blocked in the assistant's sandbox
  (`x-deny-reason: host_not_allowed` on pypi.org, despite it being an
  allowed domain — environment quirk, not a real restriction). The
  `kneed`-based module code is API-correct but not execution-verified by
  the assistant; validation numbers above came from a hand-rolled
  "max distance from chord" elbow finder (pure NumPy, mathematically
  equivalent to Kneedle for a single clean knee) as a stand-in. Worth
  confirming `curve="concave"/"convex"` choices are right once run for
  real.

## Open ideas, explicitly parked (not started)
- **Kneedle on isolated head-rising / tail-falling slopes** (not the
  whole stroke) as a way to bound segments 2/3/4 — user's framing: "just
  initial rising and final dropping slopes, +/- epsilon (arc-length
  fraction)." Single-knee case, tractable; the *middle* segmentation
  (2/3/4 boundaries generally) was explicitly descoped as a bigger,
  multi-knee problem not currently wanted.
- **Derive segment/corner count from curvature** (`K` local minima/maxima
  count) as a kinematic proxy for what `kvg:type` currently gives for
  free — motivated by the stated end-goal that `kvg:type` is scaffolding
  to be ditched eventually. Same open problem as everywhere else this
  session: distinguishing a real corner from noise wobble needs an
  amplitude threshold, not yet designed. One promising n=1 data point
  (貝-1's single clean `K` spike matching `HZ`'s one-corner name) but far
  from validated.
- **Segment via change-point on pressure/kinematics** (5-phase framing:
  pen-down transient / initial trajectory / stroke body (may contain
  ore/hane) / final trajectory / pen-up transient) — explicitly narrowed
  by the user to *only* the 4/5 boundary (tail-side), since a validated
  head-cut detector (`CJK_STROKE_H`) already exists and segment 4 is
  allowed to extend all the way back to `t[0]`. This reframing is what
  led directly to the CPD work above.

## Immediate next steps (as of session end)
1. Re-validate `pressure_knee.py`'s clamped split against the same 4
   known strokes (貝-1, 足-1, 音-6, 石-3) to confirm the `min_window` clamp
   didn't quietly break the 4/4 tail_knee lower-bound result.
2. If it holds, wire `tail_knee` into `CJK_STROKE_HZ` as the CPD sweep's
   `start_search_idx`, replacing the fixed `fraction=0.85` — directly
   targets the 石-style failure mode with a per-stroke adaptive bound
   instead of a fixed percentage.
3. Trace one of the current ±5 tied worst-cases (口, 白, 目, 草) the same
   way 石/学-4/足-6 were traced, if further accuracy is wanted — no
   shared cause identified yet, unlike the last few rounds of bugs.
4. Decide whether/when to build real head-cut detection for HZ (currently
   preloaded from CSV in the dev harness, not actually detected) — same
   caveat as the original `CJK_STROKE_HG` stub: `"cuts" in stroke.props`
   guard works, but nothing computes a real head_cut yet for this type.
5. Longer-term, revisit `default_detector`'s unconditional
   `force=True` overwrite — currently inert only because
   `cut_the_crap.py` pre-filters to `stroke_type == "HZ"` before running
   the pipeline; would silently clobber preloaded ground-truth cuts for
   any other type (e.g. the untouched compound types HZG/SW/HP/T/HZWG/PZ/
   SP/PD) if ever run unfiltered.