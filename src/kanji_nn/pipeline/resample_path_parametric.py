import numpy as np

def resample_path_parametric(stroke, error=1e-5):
    """
    Resamples the path using raw parametric steps.
    NOTE: Vertices will NOT be perfectly equidistant by distance.
    Sampling happens in the parametric domain, not the spatial domain.
    This parameterization is non-linear with respect to distance for
    most complex curves.
    """
    path = stroke.sticky["path"]
    ts = np.linspace(0.0, 1.0, stroke.n_points)

    # Pull points directly from the main path object
    vertices = np.zeros((stroke.n_points, 2), dtype=np.float64)
    for i, t in enumerate(ts):
        raw_point = path.point(t, error)
        vertices[i, 0] = raw_point.real
        vertices[i, 1] = raw_point.imag

    return stroke.clone(props={"path:xy": vertices})
