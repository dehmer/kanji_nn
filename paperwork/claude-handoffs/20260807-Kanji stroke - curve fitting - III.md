# Handoff — Bézier Fitting Project: Leg-Agreement, A-Case Repair, Dedup Discovery

Session focus: built the leg-comparison/reverse-tangent-agreement test (previously the
biggest gap), validated it against ア/0, implemented and ran A-case repair — which then
surfaced a real, previously-unknown data-quality problem (duplicate consecutive raw
points) that needs fixing *before* fitting work continues. This doc captures what's
built, what's confirmed, and the dedup work queued for next session.

**Stance carried forward, still valid:** fitting itself remains reference-blind. Corners
are still genuine G0 discontinuities. Resampling before segmentation is still off the
table (§2 of prior handoff) — the *new* dedup work (§5 below) is a data-quality fix, not
resampling; see §5 for why that distinction matters.

---

## 1. Terminology / reframing since last session

- **Traceback detection reframed as cluster detection.** `detect_tracebacks.py` →
  `detect_clusters.py`. What's detected is now explicitly just "vertex clusters in slow
  regions with relatively high turning-angle peaks" — candidates for possible cleanup,
  *not* asserted to be genuine direction reversals. Classification (is this really A/N/O,
  or an ordinary corner that slipped through) is a deferred, separate downstream step.
- **D-case retired, replaced by O-case** (user's naming convention: name cases after the
  character where first discovered). O-case = two or more angle peaks whose prominence
  clears threshold, sharing one physical speed valley. Confirmed on オ/1: visually 4
  angle spikes in one valley, only the outer 2 clear prominence.
- Corpus-wide peak scan (all strokes, `distance=1`) confirms オ/1 is currently a
  **singleton** for genuine 2-peaks-one-valley — checked る/0, そ/0, ろ/0, ね/1 by hand
  against their valley/speed plots; all four are actually *ordinary multiple distinct
  corners* (separate valleys, full speed recovery between, one clean spike each), not
  O-cases. No other O-case confirmed yet.
- `intersect_lines`/apex-from-line-intersection renamed conceptually from "corrected
  apex" to **`designated_apex`** in code — same meaning (§4 of prior handoff: intersection
  of clean entry/exit heading lines), name changed to stay neutral until classification
  exists (a cluster isn't necessarily A-case yet, so "corrected" presumed too much).

## 2. §4 (A-case cleanup) — status update

**§4 from the prior handoff no longer fully holds.** The apex-repositioning design was
right in spirit and validated once (see §3), but the *leg-rebuilding* mechanics as
originally specified (position each new point at its original fractional arc-length
progress, projected onto the new straight leg) turned out to reproduce a data artifact
rather than clean geometry — see §4 below. The apex-finding half of §4 stands; the
leg-repopulation half needs revision pending the dedup fix.

## 3. Leg-comparison test — built and validated

`detect_clusters.py` now contains the full pipeline for a candidate region:

- `find_legs(stroke, peaksfn)` — same non-destructive valley-expansion sweep as before
  (`step`/`expand`, gap-jumping over noise-floor jitter is a **deliberate feature**, not
  a bug — confirmed by user). Returns `{(i_in, i_out): [peak_indices]}`.
- `estimate_leg_heading(stroke, start, end)` — TLS/PCA principal axis over `stroke.xy`
  (raw, **not** `gauss:xy` — smoothing already happens implicitly via the window fit
  itself; adding derivative-motivated smoothing on top would double-smooth for no
  demonstrated benefit, per §2.2 of prior handoff re: central_speed). Orients the axis
  via net displacement sign to resolve PCA's 180° ambiguity.
- `intersect_lines(p1, d1, p2, d2)` — closed-form 2D line intersection via
  `np.linalg.solve`; returns `None` on near-parallel (fallback strategy — e.g.
  nearest-point-between — not yet implemented, hasn't been needed yet).
- `compute_leg_agreement(stroke, leg_bounds, leg_window=16)` — for each candidate,
  estimates clean entry/exit headings from windows just outside `(i_in, i_out)`,
  computes `signed_angle` via `atan2(cross, dot)` (chosen over `arccos(dot)` for numerical
  stability near ±180°, the worst-conditioned case for `arccos`), and computes
  `designated_apex` via `intersect_lines` anchored at `xy[i_in]`/`xy[i_out]`.

**Known gap, not yet triggered in practice:** if `i_in`/`i_out` sits very close to a
stroke boundary, `entry_start`/`exit_end` can collapse the window to zero rows, and
`estimate_leg_heading`'s SVD then has nothing to extract a principal axis from → crash.
Root-caused (traced to a missed early ん-boundary peak that lowering prominence exposed);
fixed *for that instance* by improving trim, but the underlying **no-guard-rail** issue
in `compute_leg_agreement` is still open — no code-level guard exists yet for
insufficient clean-window room. Revisit if it recurs.

### Validated results
| Case | bounds | peaks | signed_angle_deg | Read |
|---|---|---|---|---|
| ん/0 | (78,80) | [79] | 173.8° | Near-perfect anti-parallel — clean N-case signature, most unambiguous of the three |
| ア/0 | (41,53)→(41→44→53 refined) | [44] | 144.4° | Matches user's independent hand calc (148.8°) closely — method reproduces manual result |
| オ/1 | (83,134) | [86,130] | 122.1° | Weaker signal explained: bounds span *both* sub-peaks of an O-case cluster; outer-boundary heading nets out two internal reversals rather than isolating either — not a method failure, a scope mismatch (outer bounds test "is something opposed happening at all", can't characterize internal structure) |

### ア/0 apex deep-dive (settled)
- `designated_apex` landed at ≈idx38 position — *before* `i_in`(41), i.e. outside the
  detected valley bounds entirely.
- Initially read as ambiguous/concerning; resolved via user's kinematic argument:
  slowness starting near idx38 is the **expected leading edge of an honest sharp turn**,
  not contamination — the near-zero valley core (`i_in`-`i_out`) is the only region
  confidently identifiable as chaotic; clean data exists on both sides of it, and the
  designated apex landing just outside that core, near where deceleration into the turn
  begins, is the *expected*, not surprising, result.
- Same logic applies symmetrically at `i_out`(53): speed still low near idx54 doesn't
  imply exit-window contamination, since slow-but-directionally-consistent is normal
  turn behavior, not corruption bleeding into the clean window.
- **Conclusion: apex-via-line-intersection works correctly for ア/0.** This the first
  fully validated concrete case per next-session-checklist item 5 (prior handoff).

## 4. A-case repair — implemented, revealed a real bug (not in the repair logic itself)

`cleanup_clusters.py` implements the ア/0-shaped repair path:
- `repair_leg_pair(xy, i_in, peak, i_out, apex)` — discards samples strictly between
  `i_in`/`i_out`, rebuilds as two straight legs (`i_in→apex`, `apex→i_out`), repopulating
  each leg by mapping each discarded point's original fractional arc-length progress onto
  the new straight geometry. This directly implements §4 of the prior handoff.
- `cleanup_clusters(stroke)` — guarded to only handle the single-cluster, single-peak
  case (ア/0-shaped); everything else deferred until classification exists, per user's
  explicit framing this session ("classification deferred until a new case differs from
  ア/0").
- Repaired stroke returned as a **fresh `Stroke(...)`** (not `stroke.clone(...)`) — user's
  explicit correction. `clone()` only merges `features`/`props`/`sticky`, it has no path
  for rewriting `raw`. Constructing fresh also cleanly drops all stale derived
  features/props from before the repair, rather than requiring a rule to remember they're
  invalid — correctness by construction instead of by discipline.

### The discovery: repaired geometry produces a staircase artifact
Re-running the pipeline on the repaired ア/0 stroke produced **new, unexplained angle
spikes** inside/near the just-repaired region, rather than the expected clean corner.
Root cause, worked out this session:

1. The **original raw capture** contains runs of 3-4 consecutive samples with **literally
   identical `(x, y)`** — pen recorded at the same coordinate across several consecutive
   timestamps (near-zero-velocity digitizer quantization is the likely mechanism, not yet
   independently confirmed).
2. `arc_length_fractions` correctly reports zero fractional progress across these runs
   (accurate: those points genuinely didn't move) — visible as flat plateaus in the
   fraction array (e.g. `[0,0,0,0, 0.5,0.5,0.5, 1,1,1]`).
3. `repair_leg_pair` **projects those plateaus onto the new straight leg**, faithfully
   reproducing the same "stopped-stopped-stopped-teleport" pattern on otherwise-clean
   geometry — multiple new points collapse onto identical new coordinates, then jump
   discretely to the next value.
4. That staircase is numerically the same near-zero-then-instant-displacement degeneracy
   `arc_length.py`'s epsilon-nudge exists to guard against (prior handoff §1) — except
   here it's newly *introduced* into repaired data rather than pre-existing in raw, and
   nothing in the repair path guards against it.

**This is one phenomenon surfacing twice, not two independent bugs**: zero net
displacement (in the fractions) is the accurate mathematical symptom of duplicate raw
samples; the staircase (in the repaired output) is what happens when that accurate
symptom gets taken at face value and re-projected as if each duplicate-timestamp sample
deserved its own distinct position on the new leg.

**Consequence: §4's leg-repopulation recipe, as literally specified, is unsafe applied
to raw data containing duplicate-position runs — which is data this pipeline is now
confirmed to contain.** The fidelity-to-original-dwell-pattern goal behind §4 remains
right in spirit; the mechanism needs to route through deduplicated positions, not raw
`xy` directly, or it will keep re-manufacturing this artifact on future repairs.

## 5. Dedup — root-cause fix, queued for next session (not started)

Decided this session, not yet implemented:

- **Timestamps and pressure are retained in `raw`, unconditionally.** Central speed
  (used heavily, no alternative identified — acceleration, tangential acceleration,
  backward/forward speed, and pressure-derivative are all currently unused) genuinely
  requires elapsed time per sample; dropping `t` isn't viable. This was explicitly
  considered and rejected this session.
- **Approach, chosen for simplicity over array-shrinking:** collapse each run of
  (near-)duplicate consecutive `xy` positions to one representative position (first point,
  or run centroid — not yet decided which), applied **in place to `x`/`y` only**.
  `t`/`pressure` stay untouched, one row per original sample — `n_points` does **not**
  shrink. User is explicitly indifferent to array length/vertex count; this option was
  preferred purely for being the simplest change (no reshaping, no new derived feature,
  no timestamp-survivor decision to make).
  - Explicitly considered and set aside: (a) full row-collapse (reduces `n_points`,
    needs a decision about which/how timestamps survive or a new `dwell_ms` feature to
    avoid losing dwell-duration information — more machinery for a benefit user doesn't
    care about); (b) RDP or similar simplification pre-fitting — explicitly does **not**
    solve this, since RDP's perpendicular-distance test is undefined/degenerate exactly
    when neighboring points already coincide; RDP assumes distinct input positions and
    was never meant to detect same-point-sampled-N-times.
  - This is **not** a form of resampling in the sense §2 of the prior handoff ruled out —
    it doesn't change point *density* as a function of speed/arc-length, it only removes
    literal position duplicates. The speed-dependent density signal (§2 point 1 of prior
    handoff) is preserved; only degenerate zero-length runs are flattened.
- **Open question, not resolved:** exact-equality dedup only, or epsilon-based
  near-duplicate merging? Likely device/quantization-driven (sub-pixel rounding at
  near-zero velocity), which suggests near-duplicates a fraction of a unit apart could
  reproduce the same staircase pathology without being bit-identical. Needs a decision
  before implementation, not just after a second failure surfaces it.
- **Where in the pipeline:** first stage, before `arc_length` — a data-quality fix, not
  a pipeline decision that belongs downstream. This is a **destructive, unconditional**
  rewrite of raw `xy` for every stroke at ingestion, a bigger step than any prior work in
  this project — until now, raw was only ever *read* to flag candidate regions; nothing
  touched it unconditionally for every stroke. Worth being deliberate that after this,
  "raw" downstream means "raw as delivered post-dedup", not "raw as originally captured".
- **Known knock-on effects, not yet addressed:**
  - Any indices hardcoded from earlier sessions (モ/1 idx41/45, 立/2 idx16, 車/6 idx≈6,
    気/3 idx≈2, etc.) will shift post-dedup and need re-deriving — expected, one-time.
  - Every sample-count-based tuned constant (angle `prominence`/`distance`,
    `leg_window`, `find_legs`' `window`/`k`) was implicitly tuned against slightly
    duplicate-inflated point counts; re-verify against deduped data rather than assuming
    they still hold.
  - The A-case repair path (§4 above) should be re-tested against ア/0 post-dedup — the
    staircase should disappear once `repair_leg_pair` operates on already-deduplicated
    `xy`, but this hasn't been confirmed yet since dedup isn't built.

---

## 6. Immediate next-session checklist

1. Implement raw `xy` dedup as the pipeline's first stage (§5): collapse
   (near-)duplicate consecutive positions in place, `t`/`pressure` untouched,
   `n_points` unchanged. Decide exact-vs-epsilon equality before writing it, not after.
2. Decide representative-position rule for a collapsed run: first point vs. centroid.
3. Re-run ア/0 end-to-end (dedup → existing pipeline → cluster detect → repair) and
   confirm the staircase artifact is gone — this is the direct regression test for §4.
4. Re-verify all sample-count-tuned constants (angle prominence/distance, `leg_window`,
   `find_legs` window/k) against deduped data; re-tune if the corpus-wide candidate list
   changes materially.
5. Re-check the two known trim stragglers from the prior handoff (車/6 idx≈6, 気/3
   idx≈2) — indices will have shifted or the issues may already be resolved incidentally
   by dedup; re-derive fresh rather than assuming old indices still apply.
6. Add a guard in `compute_leg_agreement` for the zero/negative clean-window-length case
   at stroke boundaries (currently only avoided by chance via improved trimming, not
   actually guarded in code).
7. Decide `intersect_lines`' near-parallel fallback (nearest-point-between, per original
   §4 wording) — still unimplemented, `None` is currently the only behavior.
8. Once dedup is confirmed to fix ア/0: generalize `cleanup_clusters` beyond the
   single-cluster/single-peak guard, and begin real classification (ordinary corner vs.
   A vs. N vs. O) using `signed_angle`/scale, per the deferred-classification framing
   adopted this session.
9. O-case apex selection is still just a stated idea, not designed: incoming/outgoing
   tangents per sub-peak, see where they meet — revisit once a second O-case specimen
   turns up (corpus scan this session found none besides オ/1).