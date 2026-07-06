import numpy as np

factor = 1.0
def real(p): return p.real / factor
def imag(p): return p.imag / factor

def sample_segment_adaptively(segment, segment_length, base_density=0.1, alpha=5.0, resolution=500):
    """
    Samples an svg.path segment adaptively based on its local curvature.

    Parameters:
    -----------
    segment : svg.path.Segment
        The parsed path segment (e.g., CubicBezier).
    segment_length :
    base_density : float (default 0.1)
        Target baseline points per unit of arc length.
    alpha : float (default 5)
        Sensitivity factor for curvature. Higher means more points clustered at corners.
    resolution : int
        Size of the internal fine grid used to analyze the curve profile.
    """

    # print('base_density: ', base_density)
    # print('alpha:        ', alpha)
    # print('resolution:   ', resolution)


    # 1. Determine point budget based on arc length
    arc_length = segment_length
    budget = max(4, int(np.ceil(arc_length * base_density)))
    print('budget', budget)

    # 2. Generate a fine, uniform parametric grid to probe features
    t_fine = np.linspace(0, 1, resolution)
    h = 1e-4

    # Vectorize point extraction from svg.path using complex coordinates
    # Note: segment.point(t) takes a scalar, so we evaluate it across our grid
    p_mid = np.array([segment.point(t) for t in t_fine])
    p_min = np.array([segment.point(max(0.0, t - h)) for t in t_fine])
    p_plu = np.array([segment.point(min(1.0, t + h)) for t in t_fine])

    # Extract real (x) and imaginary (y) components
    x, y = real(p_mid), imag(p_mid)

    # 3. Approximate numerical derivatives via finite differences
    dt = 2 * h  # exact difference window for internal points
    dx = (real(p_plu) - real(p_min)) / dt
    dy = (imag(p_plu) - imag(p_min)) / dt

    dx2 = (real(p_plu) - 2*x + real(p_min)) / (h**2)
    dy2 = (imag(p_plu) - 2*y + imag(p_min)) / (h**2)

    # 4. Calculate continuous velocity (speed) and curvature
    speed = np.sqrt(dx**2 + dy**2)

    # Avoid runtime division-by-zero warnings on stationary points
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.abs(dx * dy2 - dy * dx2) / (speed ** 3)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Build the sampling density weight array
    # W represents how "important" each localized pocket of the curve is
    weights = 1.0 + alpha * curvature

    # Integrate weights across the fine grid to build a CDF
    # Using trapezoidal integration across our parametric grid
    cdf = np.zeros_like(t_fine)
    cdf[1:] = np.cumsum(0.5 * (weights[:-1] + weights[1:]) * np.diff(t_fine))

    # Normalize the CDF to map perfectly from [0, 1] to [0, 1]
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    else:
        cdf = t_fine # Fallback to uniform if weights collapse

    # 6. Interpolate down to our target budget size
    # We want uniform steps in "weight space" mapped back to physical 't' values
    target_weights = np.linspace(0, 1, budget)
    adaptive_t = np.interp(target_weights, cdf, t_fine)

    # 7. Collect final physical points and their associated features
    # This structure mirrors exactly what the LSTM sequence expects
    sampled_points = []
    for t_val in adaptive_t:
        p = segment.point(t_val)
        # Re-evaluate localized curvature at the exact chosen point for your LSTM feature channel
        feat = p_mid[np.abs(t_fine - t_val).argmin()] # Quick nearest-neighbor feature lookup
        sampled_points.append({
            'x': real(p),
            'y': imag(p),
            't': t_val
        })

    return sampled_points
