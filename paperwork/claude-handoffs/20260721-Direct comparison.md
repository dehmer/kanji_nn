# Handoff: Stroke Head/Tail Trimming — "Direct Comparison" Approach

## Project context
Collecting handwritten Kana/Kanji stroke data via a hand-rolled Flutter/iPadOS app
("painting by numbers": user traces over a KanjiVG SVG backdrop with Apple Pencil).
Raw pointer data is noisy — especially at stroke head/tail — due to a mix of:
- digitizer/tech-stack quantization (staircase artifacts in slow-speed regions)
- physical pen slip (slippery display, low friction)
- non-native handwriting habits introducing occasional hesitation/overshoot

Goal: a robust method to trim dirty head/tail regions from raw strokes, independent
of the original LSTM classification project this data ultimately feeds.

Two existing heuristic detectors already tried:
1. `ds`-minima based (simple arc-length-delta minima) — decent on simple strokes, weak on strokes with deliberate corners/hooks.
2. Change-point detection (CPD) on a weighted combination of z-scored channels — decent on tails, prone to overfitting (weights tuned on too few samples), and demonstrably wrong on at least one case (see Findings below).

## Data available
- Raw stroke arrays: `(n, 5)` = `[timestamp, x, y, pressure, pen-down/up]`.
- Derived feature pipeline (functional, chainable, each fn returns `stroke.clone(features={...})`):
  arc length (`ds`, `s`, smoothed variants), forward/backward/central speed, curvature (`K`, `θ`, `dθ/ds`),
  tangent (`tx`,`ty`), local straightness (`loc_stness`), global straightness (`stness` — low-value, see caveats),
  pressure (`P`, `P_norm`, `P_inv`, `dP`, `dP/dt`), accelerations (`am`, `at`, `ax`,`ay`), `cpd_signal` (CPD-specific, not reused elsewhere).
- `expected-cuts.csv`: per-stroke `head_cut` / `tail_cut` sample-index annotations (~100+ strokes).
  - `head_cut = 0` = **not yet annotated** (placeholder). Real head cuts always `> 0`.
  - `tail_cut = n_points` = **meaningful**: clean taper, no dirty tail at all.
  - Annotations are admittedly "good guesses," not rigorous ground truth (eyeballed).
- `types.csv`: KanjiVG `kvg_type` per stroke (Kanji only, not Kana). Confirmed via the Unicode
  CJK Strokes naming convention: name length `n` = exactly `n-1` turning points, read in stroke
  order; letters are pinyin-initial codes for stroke segments (H=héng, S=shù, P=piě, D=diǎn, N=nà,
  Z=zhé=sharp corner/"ore", W=wān=soft curve, G=gōu=hook/"hane", T=tí). `G` only ever appears as the
  **final** letter — confirmed structurally tied to the terminal hook, never mid-stroke.
- KanjiVG reference paths: confirmed (via project docs + visual side-by-side check) to be built from
  actual schoolbook font outlines, drawn through the **complete** stroke shape including hooks —
  not a simplified skeleton. Paths are stored as SVG Bézier data; a DB also holds a resampled version
  used at collection time for a DTW-based quality gate (only strokes below an aggregate DTW error
  threshold vs. reference were committed to the dataset).

## Stroke phase model (working hypothesis, still partially open)
1. Head dirty phase (explicit cut region)
2. Initial trajectory established, deliberate writing ("rise")
3. Stable stroke body (may contain zero+ corners — Z/W type turns)
4. Home stretch / final trajectory ("fall") — may contain zero or one hook (G / hane)
5. Tail noise (explicit cut region, can be multiple samples, or empty if clean exit)

Open question: whether phases 2+3 should be merged (no independently corroborated emission
signature found for a 2/3 boundary in examples checked so far — evidence points toward merge,
but only checked on a couple of stroke instances, not systematically).

## Key findings so far (apply regardless of method chosen)
- **Annotation bias, head side**: fast/committed initial strokes can get over-cut by eyeballing —
  kinematic channels (esp. `loc_stness`) showed a real head-cut label sitting ~350ms too late on
  one `SW`-type stroke; features themselves argued for an earlier, correct cut.
- **Annotation bias, tail side (mirror case)**: terminal hooks (`...G` types) risk being mistaken
  for tail artifact and over-cut — both CPD and the human annotation cut before a `HZWG` stroke's
  tail actually ended; all kinematic channels stayed "clean" well past the labeled cut, and the
  raw pen-down/up flag would resolve this unambiguously if checked.
- **`K` (curvature) numerical issue**: computed as `tx*dty - ty*dtx` divided implicitly by arc-length
  step; blows up (`~1e11` seen) when the pen is nearly stationary — which happens both in artifact
  jitter *and* at genuine sharp corners (a real turn requires deceleration). Needs a denominator
  floor or speed-gating before use in any downstream method. Also observed: no clear `K` signature
  at a suspected real hook in one `HZWG` example — unresolved, possibly a sampling/resolution issue.
- **Quantization/staircase noise concentrates at low speed** — confirmed as a known, named
  phenomenon in digital-ink literature (motor tremor + digitization noise compounding, especially on
  handheld devices; separately, "pen slip" on frictionless surfaces is its own studied problem with
  dedicated smoothing-spline mitigations). This is structurally the same low-speed regime where real
  corners occur, so a single fixed smoothing sigma can't cleanly separate the two — motivates
  speed-adaptive smoothing over one global sigma.
- **`stness`** (global straightness) is ~1 by construction (unit tangent vector sanity check), low
  information for state/segment discrimination — prefer `loc_stness`.
- Never mix raw and smoothed-derived features in the same analysis; keep consistent smoothing level
  across all channels feeding any single method.

## New direction: Direct comparison against reference path
Rather than inferring artifact regions purely from the stroke's own kinematics (self-referential),
compare each raw handwritten point directly against the known-correct KanjiVG reference geometry
(externally referenced). Hypothesis: head/tail overshoot should show up as clean, large spatial
distance from the reference path near the stroke's ends, without needing to disambiguate
corners/hooks from noise the way purely kinematic signals require — since real corners/hooks are
themselves part of the reference geometry and should show *low* distance.

### Proposed method
1. Densely resample the reference Bézier path (finer than the raw stroke's point count).
2. Compute per-point distance from raw stroke points to the reference, using a
   **monotonic-constrained** nearest-point match (not free 1-NN) — avoids false low-distance
   matches where a stroke curls close to another part of itself (e.g. near a tight turn).
3. Validate directly against the existing `head_cut`/`tail_cut` annotations across the ~100+
   labeled strokes — check whether a distance threshold crossing lines up with labels.
4. Quantitatively compare against the two existing detectors (`ds`-minima, CPD) on the same set.
5. Sanity check on a stroke already believed to have a clean tail (no artifact) — distance profile
   should stay low/flat through the true end, not spike.

### Things to confirm early
- Coordinate alignment between the capture canvas and the reference SVG frame (scale/origin/rotation) —
  probably guaranteed by the "painting by numbers" tracing setup, but worth a direct check on one sample.
- Whether per-point DTW correspondence/residuals from the original collection-time quality gate
  still exist anywhere (answer so far: no — only the raw pointer events and the DB-stored reference
  paths are available; nothing precomputed).

### Relationship to HMM (parked, not abandoned)
HMM approach (5-phase or merged 4-phase left-to-right topology, Gaussian/mixture emissions per
state, semi-supervised Baum-Welch using head/tail labels as hard-clamped boundary constraints with
the interior left latent) is set aside as a fallback, not pursued further right now. If direct
comparison has systematic gaps on certain stroke types, HMM could be revisited with the
monotonic point-to-reference distance itself as one additional input feature/channel, alongside
the kinematic channels already discussed.