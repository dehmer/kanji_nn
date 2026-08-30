import numpy as np

def resample_splines(splines: Splines, ds: float = 5.0, steps: int = 50) -> np.ndarray:
    """
    Resample each spline in `splines` (N,9) at arc-length spacing ~ds,
    using `steps` fine samples per curve as the arc-length proxy.
    Returns (M,4) xysp: x, y, segment_idx, pen.
    segment_idx: 0 for the spline's first point (Move proxy), else the
    1-based original CubicBezier index. Last point per spline: pen=0.
    """
    split_indices = np.where(splines[:, -1] == 0)[0] + 1
    spline_list = np.split(splines[:, :8], split_indices[:-1])

    t_grid = np.linspace(0.0, 1.0, steps, endpoint=False)
    out_chunks = []

    for spline in spline_list:
        K = spline.shape[0]
        p0, p1, p2, p3 = spline[:, 0:2], spline[:, 2:4], spline[:, 4:6], spline[:, 6:8]

        t = t_grid[None, :, None]
        mt = 1.0 - t
        pts = (mt**3 * p0[:, None, :] + 3*mt**2*t * p1[:, None, :]
               + 3*mt*t**2 * p2[:, None, :] + t**3 * p3[:, None, :])  # (K, steps, 2)

        fine_xy = pts.reshape(K * steps, 2)
        fine_seg = np.repeat(np.arange(1, K + 1), steps)

        fine_xy = np.vstack([fine_xy, p3[-1:]])   # true endpoint, t=1 of last curve
        fine_seg = np.append(fine_seg, K)
        fine_seg[0] = 0                            # Move proxy

        deltas = np.linalg.norm(np.diff(fine_xy, axis=0), axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(deltas)])
        total_len = cum_len[-1]

        targets = np.arange(0.0, total_len, ds)
        if targets[-1] != total_len:
            targets = np.append(targets, total_len)

        rx = np.interp(targets, cum_len, fine_xy[:, 0])
        ry = np.interp(targets, cum_len, fine_xy[:, 1])

        idx = np.clip(np.searchsorted(cum_len, targets, side="right") - 1, 0, len(fine_seg) - 1)
        rseg = fine_seg[idx]

        pen = np.ones(len(targets))
        pen[-1] = 0.0

        out_chunks.append(np.column_stack([rx, ry, rseg, pen]))

    return np.vstack(out_chunks)
