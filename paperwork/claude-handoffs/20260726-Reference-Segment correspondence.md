# Handoff: Reference-Segment Correspondence & Conflict Resolution

## Goal (refined this session)

Terminology now settled, replacing looser language from the prior handoff
("resemble the reference in curviness/smoothness"):

Handwritten strokes should be denoised so their **segment structure** —
which spans are straight, which are curved, where corners fall — is
*clarified* using the authored KanjiVG reference as a structural template,
while the actual geometric parameters of each segment (curvature
magnitude, corner angle, length, position) remain entirely the sample's
own. "Segment structure" (not "grammar," not "topology") is the preferred
term going forward:

- **Topology**: connectivity/crossings only (self-intersections, open vs.
  closed). Coarsest level, rarely relevant except for the ね-style
  self-crossing-loop case.
- **Segment structure**: the straight/curve/corner sequence along a
  stroke's arc length. This is what almost all the work targets.

**Hard constraint, restated and unshaken**: the KanjiVG reference may
*classify/label* (straight vs. curved, segment boundaries, fit-family
choice) but must **never** supply coordinates the fitted output is pulled
toward — not by snapping, not by blending, not by "merging" reference and
sample control points. This came up twice this session in different
vocabulary ("move vertices to the right spot," "merge current position
with segment Bézier definition") — both were caught and clarified before
becoming code. **Watch for this constraint eroding via rephrasing, not
just via explicit violation** — it survives restatement, not just
original phrasing.

## Where we ended up

The "easy road" decision (authored KanjiVG Bézier paths as reference
geometry, not curvature/RDP on the `wkb` polyline) has paid off. This
session built and validated the full path from raw Bézier data to a
concrete, data-backed conflict-resolution rule.

### `vg_trace_align` vs. new `dtw()` — resolved, not redundant

Two different jobs, both real:
- `vg_trace_align` (against `wkb`, pre-trim): head/tail trimming — finds
  where the real stroke starts/ends amid pen-down noise. Stays as-is,
  though may eventually be revisited to align with the SVG-Bézier
  approach — not yet done, low priority.
- `dtw()` (new, against parsed Bézier reference, end of pipeline):
  segment correspondence / labeling. Operates on already-trimmed,
  resampled, Gaussian-smoothed points (`gauss:xy`), since `dtw` runs last
  in pipeline execution order (first in `compose_pipeline`'s list, which
  executes right-to-left).

**Handoff item #4 from the prior session ("does correspondence survive
`trim_region`") is now moot** — `dtw()` runs after trim in execution
order, so there's nothing to carry through trim. Closed, not just
deferred.

### Pipeline structure (current)

```python
def compose_pipeline(wkb_reader):
    return compose(
        data.dtw,                              # segment correspondence (this session)
        data.resolve_segments,                 # NEW scaffolded stub, TODO not yet implemented
        data.turning_angle,
        metrics.arc_length_raw,
        partial(data.gauss_1d, sigma=3.0),
        data.resampling_uniform,
        metrics.arc_length_raw,
        trim_region,
        data.vg_trace_align,
        partial(data.wkb, wkb_reader=wkb_reader),
        ...
    )
```

(Actual execution order, bottom-up: trim → resample → smooth → turning-angle
→ dtw → resolve_segments.)

### `dtw.py` — cleaned up, stable

Computes DTW correspondence (`fastdtw`, `radius=1`) between `gauss:xy` and
a uniform arc-length resampling of the parsed reference Bézier path.
Produces per-warp-row tuples `(stroke_idx, ref_vertex_idx, ref_segment_idx)`
as array `D`. Detects **segment conflicts**: rows where the same stroke
index `i` appears twice with a segment index that *increases* between the
two occurrences (`rising = D[1:,2] > D[:-1,2]`, generalized — not hardcoded
to +1 steps, correctly catches any-size skips even though none have been
observed yet).

Now attaches to `stroke.props`: `D`, `conflict_rows`, plus `vlines`/`struts`
(plotting aids). **`obb_ratios`/classification was deliberately moved OUT of
`dtw.py`** — it's a property of reference geometry alone, not of the
correspondence, and belongs in the resolution step. `dtw()` is now purely
mechanical: alignment + conflict detection, no interpretation.

Known-clean idioms after cleanup pass: no dead imports, no variable
shadowing (`path` vs. warp-path collision fixed), `boundary_rows` computed
once and reused (was 3x redundant), conflict-detection loop vectorized.

### Conflict rate & characterization — the big result this session

**29 real conflicts total** across Katakana(47) + Hiragana(46) +
Kanken-10(80) = 173 strokes. All 29 share the exact same *shape*: single
stroke-index `i`, two consecutive D-rows, reference-index and segment-index
each advance by exactly 1. (The detection logic was deliberately kept
general — `y > b` not `y = b+1` — but no wider skip has ever been observed.
Treat as "not yet seen," not "impossible," same epistemic caution as the
zero-length-Bézier case below.)

**No case has ever been produced by a coincidence of `i` crossing a stroke
boundary** — D is built per-stroke, guard was discussed and isn't currently
needed since `i` conflicts have only ever been observed within a single
stroke's own D block.

### `bezier_obb` — straightness/curviness measure (built, working)

Oriented bounding box computed from Bézier control points (`p0..p3`):
chord (`p3-p0`) as major axis, perpendicular deviation of `p1`/`p2` as
minor axis. `ratio = height/width` (or similar — check `bezier_obb.py`
directly for exact formula). Superior to a naive midpoint-check
(`avg(a,b)`) because it can't be fooled by a symmetric S-curve that
happens to pass through its own chord midpoint — the OBB catches the bulge
wherever it occurs along the curve, not just at t=0.5.

**Zero-length-chord edge case**: guarded (`width_base < 1e-9` → axis-aligned
BB fallback, `ratio: None`). **This case DOES occur in real KanjiVG data**
— found twice in one character during this session (both apparently at ends
of otherwise-straight strokes; not yet confirmed whether it's systematically
an end-of-stroke authoring artifact — worth checking if it becomes relevant
again). The "KanjiVG probably doesn't have these" hunch from earlier in the
session was **wrong** — good reminder to keep the `None`-handling in place
downstream, not skip it as dead code.

### `classify_bezier.py` — NEW, adds direction on top of magnitude

Classifies each Bézier segment into `empty | near-straight | left-bend |
right-bend | s-bend`, using signed perpendicular distance of both control
points from the chord (opposite signs → s-bend; same sign → determines
left/right via net displacement). `epsilon=0.05` threshold for
near-straight, chosen by eyeballing gaps in real data (0.009–0.047 for
near-straight vs. 0.122–0.944 for bends in the actual 29-conflict set —
clean separation, no borderline cases seen yet, but not validated beyond
this eyeball check). **Sign convention was hand-verified by walking an
actual segment** (not just derived from the SVG y-down formula) — treat as
settled, no need to re-derive.

### Full characterization of all 29 conflicts (final, this session)

Every conflict now sorts cleanly into exactly one of three buckets — no
leftovers, no ambiguous middle cases:

1. **near-straight/near-straight** (6 cases: ネ/2@37, ヤ/1@41, ー/0@39,
   く/0@93, き/0@18, 休/1@47) — genuinely inconsequential, confirms your
   original hunch. Either segment claim is fine.
2. **near-straight/bend**, either order (17 cases — the majority) — assign
   the conflicting point to the **bend side**. Rationale: a straight
   segment doesn't need the boundary point for a meaningful fit; the curvy
   side benefits from the extra "wiggle room" at a slow/high-turn region.
3. **same-direction bend/bend** (6 cases: お/1@127, と/1@76, む/1@94,
   ま/2@90, ほ/3@86, は/2@88) — same-signed on both sides (both
   right-bend or both left-bend). Interpreted as **one continuous authored
   arc split across an anchor seam**, not genuine ambiguity about the
   point's nature. Resolution: **dual set-membership** — point belongs to
   both segments' point-runs for downstream fitting, undisplaced, same
   coordinate (no new geometry fabricated, no `n_points` row-count
   change). Different justification from bucket 4 below even though the
   mechanical resolution (dual membership) is the same.
4. **Opposite-direction (s-bend) conflict** — **zero observed** across all
   29 cases, despite confirmed genuine s-bend segments existing elsewhere
   in the data (e.g. え/1's own segment 3). Plausible structural reason:
   an s-bend's direction reversal is captured *inside* one segment's own
   geometry; a conflict straddling an s-bend boundary would additionally
   require the *neighboring* segment to also flip sign right at that seam
   — a rarer coincidence than an s-bend simply occurring. Treat as
   "not yet seen, plausibly rare, not impossible" (same posture as the
   zero-length-chord case before it showed up). **This is still the
   genuine-ambiguity case the design most needs to get right if/when it
   appears** — dual set-membership is the prepared fallback, no new code
   needed when it does.

### Rejected/parked idea: vertex splitting

Considered and parked (not rejected outright): instead of dual
set-membership, physically split an ambiguous vertex into two new points
linearly interpolated toward its stroke-neighbors, one assigned to each
segment. Problems: still fabricates coordinates the stylus never produced;
breaks `n_points` row alignment across `raw` and all feature columns
(circumventable — a fresh `Stroke` built directly, same pattern as
`resampling_uniform`, sidesteps `Stroke.clone()`'s alignment check — but
not worth the complexity yet); and two fits anchored to two *different*
nearby points may actually introduce a discontinuity at the boundary,
rather than fixing one, since set-membership's shared-coordinate property
is what currently guarantees the two fits meet cleanly. Parked "in the
back of the head" per user — revisit only if a concrete case demands it.

## Immediate next steps (in likely order)

1. **Flesh out `resolve_segments.py`** (currently a stub with the conflict
   print copied over from old `dtw.py`, plus unused leftover imports —
   `fastdtw`/`euclidean`/`ScaledPath` should be deleted, they don't belong
   here anymore). This is the concrete next task:
   - Reduce `D` (which has repeated/skipped stroke-indices in warp-row
     space) down to **one clean per-point representation**: for each of
     the stroke's actual `n_points` points, either a single resolved
     segment id, or — for genuine both-elevated-but-same-direction
     conflicts and any future s-bend conflicts — membership in two
     segments.
   - Apply the three-bucket rule above using `classify_bezier` output
     (need to decide: call `classify_bezier` here, per segment, once per
     stroke — replaces the old inline `bezier_obb`-only call).
   - Decide the concrete output shape: per-point segment-id array (length
     `n_points`) is the base case; conflicts need either a second parallel
     array (secondary segment id, `-1`/`None` where not applicable) or a
     small side-table of point-index → {segment_a, segment_b}. Not yet
     decided — the natural next design conversation.
2. Once `resolve_segments` produces real per-point labels: decide how/
   whether these labels should propagate through `trim_region` — same
   open question flagged for `vg_trace_align`'s correspondence in the
   prior handoff, now actually relevant since `resolve_segments` sits
   downstream of trim already, so this may also be moot (worth a quick
   check, same way item #4 turned out to be moot this session).
3. Curve fitting itself — **deliberately not designed yet**, and current
   understanding is explicitly shaky (user has no curve-fitting
   background, said so plainly this session — treat future fitting
   proposals from user with extra care for fit-target-boundary language,
   per the two near-misses this session). What's agreed so far: reference
   Bézier shape may inform *fit family/type choice* (line vs. spline,
   roughly how flexible) and possibly *initial parameter guess* for a
   solver — but fitted control points must be solved from hw points alone,
   never blended with or pulled toward reference coordinates. This
   boundary needs to be re-confirmed explicitly once real fitting code is
   discussed, not assumed to be understood from this summary alone.
4. Not yet investigated: whether the two zero-length-Bézier occurrences
   found this session are systematically an end-of-stroke authoring
   artifact (both seen were near stroke endpoints, small sample size — 2
   instances, 1 character). Only worth chasing if it recurs or starts
   affecting `resolve_segments` output.

## User working style notes (carried over + refined this session)

- 35+ years SW engineering experience; new to Python — flag idiom/gotchas,
  skip design hand-holding. (Confirmed working well this session:
  variable-shadowing, dead-import, and n_points-alignment flags were all
  well received and acted on.)
- **New this session, added to framing prompt verbatim**: user is new to
  this project's specific field (DTW/curve-fitting/Bézier geometry, not
  just Python) and explicitly values **conceptual/terminology grounding
  before technical implementation**. Actively watch for imprecise or
  inconsistent language — e.g. "resemble in curviness," "grammar" vs.
  "topology," "move vertices" vs. "merge" vs. legitimate fit-family
  selection — and surface the distinction before it becomes code. This
  was the single most load-bearing dynamic of the session; several
  hard-constraint near-misses were caught this way, not by user already
  knowing the line was there.
- Iterates in small, concrete steps with real plotted/printed data at each
  stage (D arrays, conflict lists, OBB-ratio plots, strut diagrams) rather
  than large speculative leaps. Consistently updates or reverses a stated
  hunch when data contradicts it (zero-length chords, "straight segments
  dominate conflicts" refined into three precise buckets) — comfortable
  being wrong quickly, prefers that to being cautious slowly.
- Explicitly enjoys visual/diagnostic tooling as a way to build confidence
  before committing to a rule (angle-plot vlines, strut overlays, OBB-ratio
  annotated plots) — offering to sketch this kind of throwaway diagnostic
  is usually welcome, more so than jumping straight to production code.
- Wants code only when explicitly asked or the design is settled;
  otherwise discussion/review/critique. Responds very well to direct
  pushback on terminology and on hard-constraint boundary risk
  specifically — this is not an area to soften language around.