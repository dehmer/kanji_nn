"""
Step 2 -- per-stroke trim decision.

Consumes the raw stroke arrays plus the Step 1 metrics (speed,
curvature, arclen_cum, dist_to_start, min_dist_nonadjacent) and decides,
for a single stroke:

  - whether it's a case we don't yet handle (short / closed / possibly
    self-intersecting -- dakuten/handakuten territory) -> flag it and
    do NOT attempt to trim, rather than risk mangling it;
  - otherwise, where (if anywhere) to cut at the start and/or end.

Nothing here mutates the stroke. The output is indices into the
*original* raw arrays (same convention as Step 1), so Step 3 can slice
with plain `stroke[start_cut:end_cut]`.

All thresholds live in TrimConfig, deliberately exposed rather than
hardcoded -- everything here is a heuristic derived from a handful of
manually-inspected strokes, and is expected to need retuning once run
across a real, larger sample. Treat the defaults as a starting point,
not a final answer.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


Status = Literal[
    "ok",                     # trimmed normally (start_cut/end_cut may still be None each)
    "flagged_short",          # too little stroke to safely reason about (dakuten territory)
    "flagged_selfintersect",  # closed/self-crossing stroke (handakuten territory) -- parked
    "flagged_ambiguous",      # start/end windows would overlap or consume the whole stroke
]


@dataclass
class TrimResult:
    start_cut: Optional[int]   # first index to KEEP; None == keep from sample 0
    end_cut: Optional[int]     # first index to DROP at the tail; None == keep to the last sample
                                # i.e. the kept stroke is stroke[start_cut:end_cut]
    status: Status
    detail: dict = field(default_factory=dict)  # onset indices, thresholds used, etc. -- for debugging/review


@dataclass
class TrimConfig:
    # --- guard thresholds (checked first; any hit -> flagged, no trimming attempted) ---
    min_arclen: float = 0.03        # total stroke arc length below this -> too short to reason about
    min_duration_ms: float = 80.0   # total stroke duration below this -> too short to reason about
    selfintersect_eps: float = 0.02 # min_dist_nonadjacent below this (excluding endpoints) -> possible loop/closure
    selfintersect_adjacency_guard: int = 2  # ignore self-proximity within this many samples of either end
                                             # (a stroke's true start/end sitting close together, e.g. a tight
                                             # hook, shouldn't by itself trigger the loop guard -- only an
                                             # *interior* near-self-crossing should)

    # --- windowing (start/end sized independently, absolute-time-first per our findings) ---
    start_window_ms: float = 180.0
    start_window_frac: float = 0.5   # cap window at this fraction of the stroke's own duration too
    end_window_ms: float = 180.0
    end_window_frac: float = 0.5

    # --- start-side "commit" detection ---
    start_persist_samples: int = 3   # speed+curvature must look "committed" for this many consecutive samples
    speed_rel_thresh: float = 0.25   # fraction of the stroke's main-body median speed counted as "moving for real"
    start_curv_thresh: float = 0.6   # rad; below this counts as "calm" for commit detection

    # --- end-side anomaly detection ---
    end_curv_thresh: float = 0.8     # rad; above this counts as "flagged" for end detection
    end_max_calm_gap: int = 2        # allow short calm gaps of up to this many samples without resetting
                                      # the detected onset (tolerates single noisy/clean samples)

    # --- ambiguity guard ---
    min_kept_fraction: float = 0.3   # if trimming would keep less than this fraction of the stroke, flag instead


def _longest_false_run(mask: np.ndarray) -> int:
    """Length of the longest run of False values in a boolean array."""
    if mask.size == 0:
        return 0
    longest = run = 0
    for v in mask:
        if not v:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _window_mask(t: np.ndarray, from_start: bool, window_ms: float, window_frac: float) -> np.ndarray:
    duration = t[-1] - t[0]
    cap_ms = min(window_ms, window_frac * duration) if duration > 0 else window_ms
    if from_start:
        return (t - t[0]) <= cap_ms
    else:
        return (t[-1] - t) <= cap_ms


def _check_guards(x, y, t, arclen_cum, min_dist_nonadjacent, config: TrimConfig, body_mask):
    n = len(x)
    total_arclen = arclen_cum[-1] if n else 0.0
    duration = t[-1] - t[0] if n else 0.0

    if total_arclen < config.min_arclen or duration < config.min_duration_ms:
        return "flagged_short", {"total_arclen": total_arclen, "duration_ms": duration}

    # Self-intersection is only meaningful as a "this might be a closed
    # shape" signal when it happens in the stroke's own body -- i.e.
    # outside the start/end transient windows. The start-side dither is
    # itself expected to loop back close to the stroke's own start point
    # (that's what a "backswing" is), so checking the full array here
    # would misfire on an ordinary open stroke with a start artifact,
    # not just on genuine closed/self-crossing shapes.
    guard = config.selfintersect_adjacency_guard
    candidate_mask = body_mask.copy()
    if n > 2 * guard:
        candidate_mask[:guard] = False
        candidate_mask[-guard:] = False
    else:
        candidate_mask[:] = False

    if candidate_mask.any():
        interior = min_dist_nonadjacent[candidate_mask]
        if np.nanmin(interior) < config.selfintersect_eps:
            local_i = int(np.nanargmin(interior))
            i = np.where(candidate_mask)[0][local_i]
            return "flagged_selfintersect", {"min_dist": float(interior.min()), "at_index": int(i)}

    return None, {}


def _detect_start_cut(speed, curvature, mask, config: TrimConfig, main_body_speed):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, {}

    speed_thresh = config.speed_rel_thresh * main_body_speed
    n = len(speed)
    persist = config.start_persist_samples

    for i in idx:
        j = i + persist
        if j > n:
            continue
        seg_speed = speed[i:j]
        seg_curv = curvature[i:j]
        if np.any(np.isnan(seg_speed)):
            continue
        curv_ok = np.abs(np.nan_to_num(seg_curv, nan=0.0)) < config.start_curv_thresh
        if np.all(seg_speed > speed_thresh) and np.all(curv_ok):
            return (i if i > 0 else None), {"onset": i, "speed_thresh": speed_thresh}

    # window fully covered by dithering with no confirmed commit -> can't safely say where
    # the real stroke starts; leave it to the ambiguity check downstream rather than guessing.
    return "ambiguous", {"speed_thresh": speed_thresh}


def _detect_end_cut(speed, curvature, mask, config: TrimConfig):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None, {}

    n = len(speed)
    flagged = np.abs(np.nan_to_num(curvature, nan=0.0)) > config.end_curv_thresh

    # Only an *actually flagged* sample can be the onset -- a suffix that
    # happens to be short is not itself evidence of a problem. Without
    # this, a perfectly clean stroke gets "detected" as soon as the
    # window is close enough to the end that its (entirely calm) tail
    # trivially satisfies the calm-gap tolerance.
    candidate_idx = idx[flagged[idx]]
    for i in candidate_idx:  # ascending == chronological within the window
        calm_run = _longest_false_run(flagged[i:n])
        if calm_run <= config.end_max_calm_gap:
            return (i if i < n else None), {"onset": i}

    return None, {}


def derive_trim(
    x, y, t,
    speed, curvature, arclen_cum, dist_to_start, min_dist_nonadjacent,
    config: TrimConfig = TrimConfig(),
) -> TrimResult:
    """
    Decide start/end trim indices for one stroke, or flag it as a case
    the general rule shouldn't touch.

    Parameters mirror Step 1's raw/metric arrays directly -- pass the
    columns you already have from `compute_stroke_metrics`, no
    repackaging needed.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(x)

    start_mask = _window_mask(t, True, config.start_window_ms, config.start_window_frac)
    end_mask = _window_mask(t, False, config.end_window_ms, config.end_window_frac)
    body_mask = ~(start_mask | end_mask)

    status, detail = _check_guards(x, y, t, arclen_cum, min_dist_nonadjacent, config, body_mask)
    if status is not None:
        return TrimResult(start_cut=None, end_cut=None, status=status, detail=detail)

    # main-body speed: everything outside both candidate windows, used as the
    # "this is what real, committed motion looks like in this stroke" reference
    body_speed = speed[body_mask]
    body_speed = body_speed[~np.isnan(body_speed)]
    main_body_speed = np.median(body_speed) if body_speed.size else np.nanmedian(speed)

    start_cut, start_detail = _detect_start_cut(speed, curvature, start_mask, config, main_body_speed)
    end_cut, end_detail = _detect_end_cut(speed, curvature, end_mask, config)

    if start_cut == "ambiguous":
        return TrimResult(start_cut=None, end_cut=None, status="flagged_ambiguous",
                           detail={"reason": "no confirmed commit point in start window", **start_detail})

    effective_start = start_cut or 0
    effective_end = end_cut if end_cut is not None else n
    kept_fraction = (effective_end - effective_start) / n if n else 0.0

    if effective_end <= effective_start or kept_fraction < config.min_kept_fraction:
        return TrimResult(
            start_cut=None, end_cut=None, status="flagged_ambiguous",
            detail={"reason": "trim would remove too much of the stroke",
                    "start_cut": start_cut, "end_cut": end_cut, "kept_fraction": kept_fraction},
        )

    return TrimResult(
        start_cut=start_cut, end_cut=end_cut, status="ok",
        detail={"start": start_detail, "end": end_detail, "main_body_speed": main_body_speed},
    )