import numpy as np

def densify(xy, max_ds):
    """
    Insert linearly-interpolated points into a polyline so that no
    consecutive-point gap exceeds max_ds. Original vertices (incl. corners)
    are preserved exactly -- points are only ever added, never moved or dropped.

    xy: (N, 2) array of polyline vertices, single stroke (no pen-state column)
    max_ds: target maximum spacing between consecutive points

    Returns: (M, 2) array, M >= N
    """
    out = [xy[0]]
    for p0, p1 in zip(xy[:-1], xy[1:]):
        d = np.linalg.norm(p1-p0)
        n_inserts = int(np.ceil(d / max_ds))-1
        if n_inserts > 0:
            t = np.linspace(0, 1, n_inserts+2)[1:-1]
            pts = p0+t[:, None] * (p1-p0)
            out.extend(pts)
        out.append(p1)
    return np.array(out)
