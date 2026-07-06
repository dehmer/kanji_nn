import cmath

def get_geometric_features(path_obj, t, h=1e-4):
    """
    Extracts geometric features from an svg.path object at parametric time t.
    Works for both a single Segment or an entire Path.
    """
    # Handle boundary conditions for finite differences
    t_minus = max(0.0, t - h)
    t_plus = min(1.0, t + h)

    # Get complex number points from svg.path
    p_mid = path_obj.point(t)
    p_min = path_obj.point(t_minus)
    p_plu = path_obj.point(t_plus)

    # Convert complex numbers to (x, y) vectors
    xy = (p_mid.real, p_mid.imag)

    # 1st Derivative (Central difference adjusted for boundaries)
    dt = t_plus - t_minus
    dx = (p_plu.real - p_min.real) / dt
    dy = (p_plu.imag - p_min.imag) / dt

    # 2nd Derivative
    # Standard second derivative central difference requires equal spacing
    # Near 0 or 1, we use an approximation or forward/backward fallback
    dx2 = (p_plu.real - 2*p_mid.real + p_min.real) / (h**2)
    dy2 = (p_plu.imag - 2*p_mid.imag + p_min.imag) / (h**2)

    # Calculate speed (metric coefficient) and curvature
    speed = cmath.sqrt(dx**2 + dy**2).real

    # Prevent division by zero in straight lines/dead points
    if speed < 1e-6:
        curvature = 0.0
    else:
        curvature = abs(dx * dy2 - dy * dx2) / (speed ** 3)

    return {
        "point": xy,          # (x, y)
        "speed": speed,        # v(t)
        "curvature": curvature # kappa(t)
    }
