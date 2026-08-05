# Handoff — Bézier Fitting Project: Corner / Traceback Cleanup

Session focus: before attempting any Bézier segmentation, characterize and clean up
artifacts in raw handwritten stroke data (vector, KanjiVG-templated, iPad/Apple Pencil
capture) that would otherwise corrupt corner detection. This doc captures the taxonomy,
design decisions, and open threads so a new session can pick up without re-deriving them.

**Stance carried forward from earlier sessions:** fitting itself will be **reference-blind**
(Schneider-style, data-driven). KanjiVG is only used offline, by us, to sanity-check our own
reasoning or later evaluate fit quality — never as a live input to segmentation logic.

---

## 1. Data model / pipeline architecture

- `Character` / `Stroke` — frozen dataclasses. `Stroke.raw` = `[t, x, y, pressure]` per point
  (5-col npy on disk also carries pen-down/up in col 4; last sample of each stroke = `0`,
  all others `1`).
- `Stroke.clone(features=…, props=…)` merges dicts immutably; **raises on duplicate keys**
  unless `force=True`. Row-count of any feature array is validated against `n_points`.
- Pipeline stages are pure functions `Stroke -> Stroke`, chained via `compose(...)`.
  **`compose` applies right-to-left** — e.g. `compose(tap(plot), turning_angle, arc_length)`
  runs `arc_length` first, then `turning_angle`, then the plot tap last.
- Raw `xy` is already Savitzky-Golay filtered upstream, once, generically. `arc_length`
  applies a **second**, lighter Gaussian pass (`sigma=1.0`, `mode='reflect'`) specifically
  for derivative-sensitive features (angle, curvature), stored as `gauss:xy`.

### Known files
| File | Produces | Notes |
|---|---|---|
| `arc_length.py` | `raw:ds/s/s:norm`, `gauss:xy/ds/s/s:norm` | epsilon-nudges `s` to strictly increasing; **first pipeline stage** |
| `turning_angle.py` | `angle` (windowed, signed, lag `w`, NaN at edges) | sits on `gauss:xy`; now typically used as `angle:w=1:abs` |
| `central_speed.py` | `raw:speed:central` | `np.gradient(x,t)`/`(y,t)` — proper non-uniform-grid central diff, per-point aligned, no fabricated boundary values. **Sits on raw `xy` by decision** (see §4) |
| `backward_speed.py` / `forward_speed.py` | `raw:speed:backward/forward` | same underlying interval speeds, just attributed to different indices + physically-motivated 0-padding at the touch/lift end. **Not used for corner detection** — kept for reference |
| `gauss_1d.py` | `gauss:xy` | **Redundant** with `arc_length`'s `gauss:xy` — same computation, same key. Latent `clone()` collision risk if both ever land in one `compose()` chain. Candidate for removal. |
| `detect_tracebacks.py` | evolving | see §3/§5 |

### Bug found & fixed (locally, needs porting back to real repo)
`stroke.py` dataclass field `props: dict[str, Any]` referenced `Any` without
`from typing import Any` — raises `NameError` at class-definition time, not just on use.

---

## 2. Core design decisions (settled, don't re-litigate)

1. **No resampling before segmentation.** Point density is speed-dependent (pen decelerates
   into corners → naturally denser sampling there) even though timestamps are ~time-uniform.
   This non-uniformity is *signal*, not noise — resampling (spatial or temporal) would erase
   evidence we need before we've used it. Resampling may return later for unrelated reasons
   (fixed-length LSTM/Transformer input), but that's a downstream, separate decision.
2. **`central_speed` stays on raw `xy`**, not `gauss:xy`. Tested head-to-head on ア's corner:
   negligible difference at `sigma=1.0` (corner minimum 0.0 → 2e-5, same index; flat-region
   jitter ~identical). No evidence smoothing helps here — revisit only if a noisier stroke
   actually shows divergence.
3. **Corners are modeled as genuine G0 discontinuities.** Explicit call: *"Sharp it is."* No
   per-instance smooth-vs-sharp litigation based on how KanjiVG's (uniformly smooth) reference
   renders the same bend — pen-style sharp execution is a legitimate stylistic choice
   independent of brush-style reference smoothness.
4. **Detection sweep is non-destructive.** "Do the full sweep again and ignore apex points
   already identified" — no array splicing/index bookkeeping during detection. Repair
   (actual array rewriting) happens once, at the end, using the complete set of findings
   against the still-intact original data.
5. **Interpolation (inventing new smooth points) is the wrong tool for corner cleanup** —
   would falsely smooth an intended sharp G0 discontinuity into false continuity. Only
   used for *repopulating* a corrected leg's point density (see §4), never to soften a corner.

---

## 3. Traceback taxonomy — the main output of this session

A **traceback** = a local, near-anti-parallel reversal in heading (pen doubles back over
its own recent track), *distinct from* an ordinary sharp corner (heading changes, but not
anti-parallel) and *distinct from* a loop (heading rotates steadily in one sense, never
anti-parallel locally, even though cumulative rotation is large).

Terminology adopted: **"farthest point out"** = the apex/extremum sample of a candidate
(user's term, replaces earlier "i\*"). `i_in` / `i_out` = last trustworthy sample before /
first trustworthy sample after the contaminated region.

| Case | Meaning | Scale (empirical) | Apex handling |
|---|---|---|---|
| **A** (ア) | Unintentional overshoot + correction | ~6–7 samples, ~50–100ms, ~1% of local stroke extent | **Repositionable** — reconstruct via intersection (or nearest-point-between) of the *clean* entry-heading line and *clean* exit-heading line. The raw farthest-point-out sample is systematically overshot evidence, not an unbiased vertex estimate — don't use it directly. |
| **N** (ん) | Genuine, deliberate, prolonged retrace | ~27+ samples, ~250ms+, ~24–34% of local leg length (measured on え) | **Preserved verbatim** — apex is authoritative, not corrected. Only the *noisy neighbors* (if any) get density-preserving repopulation. |
| **T** (trim-defect) | Residual pre-contact / post-liftoff motion misread as a corner | Boundary-proximity (near idx 0 or n−1); confirmed on モ/1 (tail, ~8 samples over) and 立/2 (head, ~16 samples, 3 angle spikes in one junk span) | **Truncate**, not reposition/preserve. Fixed mostly upstream by improving trimming (2 rounds done this session; see §6). |
| **D** (dwell) — *tentative, likely retracted* | Originally hypothesized for オ/1's 418ms chaotic span (angle ≈ exactly π, speed ≈ 5e-9 — numerically degenerate, arctan2 on near-zero-length vectors) | — | After retrim, オ/1 collapsed from that chaos to exactly 2 clean peaks. Best current explanation: this was **T-case contamination smearing inward** through the Gaussian window, not a genuine mid-stroke planning pause. **Not fully confirmed** — one more spot-check recommended before fully discarding D as a category. |

### Why scale, not signature, separates A from N
Both share the anti-parallel-heading signature. What differs is how far/long the reversal
runs. Measured via: fit a local line direction from stable flanking samples, project the
retrace-region points onto (along-line, perpendicular) coordinates relative to that line.
A stays in a tight perpendicular band for only a few samples; N stays tight for many more
samples and covers much more along-line distance. (Full worked example: ア vs え, in-session.)

### Why loops don't trigger this
Confirmed empirically on む: through the actual self-crossing loop, windowed turning angle
stayed within roughly ±26°, speed dipped only moderately (~0.0007) — nothing like the
near-zero crash + ~150–180° spike a real reversal produces. A separate, unrelated genuine
corner (~140°, real speed crash) exists later in the same stroke, at the bottom hook —
unrelated to the loop. **Conclusion: a plain, fairly aggressive prominence threshold on
windowed turning angle structurally cannot mistake a loop for a traceback**, regardless of
loop size — no separate scale-relative guard needed for this specific confusion.

### Still open / explicitly deferred
- **N-case cubic-fitting strategy** — how to actually fit segments to geometry that nearly
  spatially coincides with itself but runs in opposite parameter directions. Claude's
  in-session framing ("aligned in/out trajectories," "mirrored control points") was flagged
  by the user as unclear / possibly wrong. **Explicitly tabled** — revisit fresh, don't just
  resume from that framing.
- **Epsilon/windowing gate for auto-cleanup eligibility** (user's own instinct): only
  auto-apply A-case cleanup if the contaminated cluster is small/tight (in both `ds` and
  `dt`) relative to stroke scale; larger/looser clusters get flagged for manual review
  instead of auto-cleaned. Would double as a backstop against ever misfiring on loops/N-case
  even independent of the angle/speed signature. **Proposed, not formalized.** User said
  "I'm going to mull over this" — check whether that landed anywhere.

---

## 4. A-case cleanup algorithm (fully designed, not yet coded)

Given a confirmed A-case candidate with `i_in`, `farthest_point_out`, `i_out`:

1. Compute corrected apex = intersection (or nearest-point-between, if not exact) of the
   clean entry-heading line and clean exit-heading line.
2. Discard all raw samples strictly between `i_in` and `i_out`.
3. Rebuild as **two straight legs**: `i_in → corrected_apex` and `corrected_apex → i_out`.
   Split point among the *discarded* points (which ones "belong" to which leg) = the
   original farthest-point-out sample — the natural physical hinge.
4. Repopulate each leg with the **same point count** as was discarded on that side.
   - **Timestamps: carry over the original discarded points' timestamps unchanged.**
     `dt` is then exact by construction — the real elapsed time is preserved automatically,
     no time-budget arithmetic needed.
   - **Positions: place each new point at the same fractional arc-length progress** the
     original point had along its messy sub-path (0% at the leg's start, 100% at the
     apex/split), but projected onto the new *straight* leg. This transplants the real
     dwell/density pattern (slow-down into the vertex) onto the clean geometry instead of
     inventing a uniform distribution.

N-case repair uses the same leg-rebuilding mechanics, just anchored at the *unmoved*
observed apex rather than a corrected one.

T-case repair is a different primitive entirely: **truncate** at the junk/real-motion
transition point (not reposition, not preserve-and-repopulate).

None of the three repair actions are implemented in code yet — detection/classification
has been the focus so far.

---

## 5. Detection front-end — status and tuning history

Current shape of `detect_tracebacks.py`:
```
mask/peaks = find_peaks(angle:w=1:abs, prominence=…, distance=…)
paired against raw:speed:central < speed_epsilon
```

### Threshold tuning history (prominence, in radians)
| Value | Result |
|---|---|
| `0.4·π` (72°) | Far too low — flags ordinary honest corners with no traceback at all |
| `5/6·π` (150°) | 47 candidates across 104 hand-picked complex-stroke characters; clean single-katakana sanity checks (ア, ム, ワ, ラ, カ, オ each → exactly one peak) |
| `π` (180°) | Only **5** candidates — surprising drop |
| `2/3·π` (120°), post-retrim | Current working point; full-corpus run == complex-subset run (see §6) |

**Why 150°→180° dropped from 47→5, resolved:** the reversal isn't always concentrated in a
single sample-to-sample transition. `angle:w=1` only sees the rotation across one lag-1
window; if the true ~180° turn is smeared across 2–3 consecutive steps (a few degrees each,
same rotational sense), no single sample ever reports the full angle — each sees only its
slice. **Conclusion: `find_peaks` on `angle:w=1` is a *locator*, not the actual reversal
test.** It only needs to land somewhere inside the smeared-reversal neighborhood; it doesn't
need to hit the true peak value. The actual arbiter has to be the leg-comparison test (§5.1),
using wide, non-adjacent windows immune to this smearing.

### 5.1 Leg-comparison / reverse-tangent-agreement — NOT YET IMPLEMENTED
This is the single biggest remaining gap. Manually validated (by hand, not in code) on:
- **ア**: clean entry heading (samples idx20–36) ≈ −6.7°, clean exit heading (idx60–84)
  ≈ 142.1° → angle between = **148.8°**. Confirms genuine A-case even judged only from
  trustworthy flanking data, independent of the local retrace noise.
- **え**: projection/perpendicular-distance construction (fit line from stable flanking
  window, project retrace-region points onto it) — confirmed N-case scale (~24–34% of leg
  length in a tight perpendicular band over ~27+ samples), an order of magnitude beyond ア.

This needs to become real code: given a candidate region, estimate clean entry/exit
headings from windows *well outside* the contaminated span, and (a) test anti-parallel-ness
to confirm/reject "this is a traceback at all" (rejects ordinary sharp corners), then
(b) if confirmed, measure scale to classify A vs N.

### 5.2 Speed-pairing — reframed, not just "confirmation"
At genuinely high prominence (150°+), speed-near-zero is close to a **kinematic
necessity** (can't reverse direction without passing through near-zero speed) — largely
redundant with a strong angle reading there. Its real value is at the **artifact-rejection**
end: catching cases where angle spikes from quantization/sensor noise *without* any matching
deceleration — i.e. guarding against non-physical angle readings, not confirming physical
ones. Recommend re-labeling this in code comments accordingly.

### 5.3 `distance` parameter caveat — not yet stress-tested
Same non-uniform-sampling issue as `turning_angle`'s window `w`: `distance` is a fixed
*sample* count, but sample density is speed-dependent. Risk: two genuinely distinct corners
occurring close together in a slow/dense region could be merged; one smeared event in a
fast/sparse region might not be merged enough. **Recommended stress test, not yet run:**
re-check `な/0`'s original tightly-spaced hits `[4, 10, 12, 15]` (spacings 2–6) against
current `find_peaks` settings to see whether it's correctly collapsing one event or
incorrectly merging several.

---

## 6. Trimming — head/tail boundary defects (mostly resolved, verify one straggler)

Two confirmed real cases this session, both same mechanism: residual pen motion
(pre-contact hover / post-liftoff drift) just past the *true* stroke boundary produces the
same kinematic signature as a real corner (angle spike + speed valley), because it also
decelerates-and-wobbles — but it isn't writing at all.

- **モ/1 (tail)**: true end ≈ t=346ms / idx41; pipeline candidate spanned to idx45 (~8
  samples of junk past the real end).
- **立/2 (head)**: three angle spikes crammed into first ~130ms (idx 2, 8, 15) — one
  contiguous junk span before real motion starts at idx16 (~16 samples of junk).

Two rounds of trimming improvements were made in-session. After the **second** round:
- Running the full character corpus (not just the hand-picked complex-stroke subset)
  produces **the same candidate list** as the subset — good evidence trimming is now
  behaving consistently across the board, not just on cherry-picked cases.
- オ/1's previous 418ms chaotic span (see D-case, §3) collapsed to exactly 2 clean peaks.

**Known straggler, not yet confirmed fixed:**
- **車/6, idx≈6** — flagged before the most recent retrim as still escaping proper
  trimming. Status after the *latest* round is unconfirmed — check this first in the next
  session.
- **気/3, idx≈2** — flagged as the last outlier in the most recent (post-2nd-retrim) run,
  right as the session ended. Not yet inspected.

---

## 7. Immediate next-session checklist

1. Re-check `車/6` and inspect `気/3` — confirm whether these are still trim artifacts or
   genuine boundary-adjacent corners now that trimming has improved.
2. Spot-check the well-separated multi-peak strokes from the latest run (`る/0 [24,78]`,
   `ろ/0 [17,70]`, `ね/1 [15,57]`, `そ/0 [20,85,147]`, `オ/1 [87,131]`) — confirm these are
   genuine multiple distinct corners per stroke, not a residual artifact.
3. Build the leg-comparison / reverse-tangent-agreement test (§5.1) — this is the actual
   gate that turns "corner candidate" into "confirmed traceback," and the only thing that
   can currently distinguish a real traceback from an ordinary sharp corner. Everything
   downstream (A/N classification, cleanup) depends on it.
4. Stress-test `find_peaks`' `distance` parameter against `な/0`'s original tight cluster
   (§5.3).
5. Once leg-comparison exists: implement the A-case cleanup algorithm (§4) against ア as
   the first concrete test case.
6. Revisit N-case cubic-fitting strategy fresh (§3, explicitly deferred, don't reuse the
   "aligned in/out trajectories" framing without re-deriving it).
7. Decide on the epsilon/windowing auto-cleanup gate (§3) — still an open "mulling" item
   from the user.
8. Consider removing `gauss_1d.py` given `arc_length.py` already produces `gauss:xy` under
   the same key — currently just a latent collision risk sitting in the codebase.