import numpy as np
from svg.path import CubicBezier


def sample_path(path, ds=0.006):
    steps = 50 # steps per segment

    points = []
    indices = []
    for seg_idx, seg in enumerate(path):
        seg_t = np.linspace(0, 1, steps)

        # Drop first t value for all segments but the first (p3 == q0):
        if seg_idx > 0:
            seg_t = seg_t[1:]

        seg_points = [seg.point(t) for t in seg_t]
        seg_points = np.array([[p.real, p.imag] for p in seg_points])
        seg_indices = np.full(len(seg_points), seg_idx)

        points.extend(seg_points)
        indices.extend(seg_indices)


    points = np.vstack(points)
    deltas = np.diff(points, axis=0)
    step_lengths = np.sqrt(np.sum(deltas**2, axis=1))
    cum_distances = np.concatenate(([0.0], np.cumsum(step_lengths)))
    total_length = cum_distances[-1]
    target_distances = np.arange(0, total_length, ds)

    # Explicitly append the exact final point if it falls short of ds
    if len(target_distances) == 0 or target_distances[-1] < total_length:
        target_distances = np.append(target_distances, total_length)

    # Interpolate X, Y coordinates and Segment Indices instantly
    x = np.interp(target_distances, cum_distances, points[:, 0])
    y = np.interp(target_distances, cum_distances, points[:, 1])

    # Use 'nearest' mapping for segment index to keep them as discrete integers
    s = np.interp(target_distances, cum_distances, indices)
    s = np.round(s) # Ensure clean rounding to closest segment ID

    return np.column_stack((x, y, s))


path_fn = lambda s: s.sticky["path"]


def resample_fixed_distance(stroke, path_fn=path_fn, ds=0.006):
    path = path_fn(stroke)
    xys = sample_path(path, ds)
    return stroke.clone(props={"path:xys": xys}, force=True)
