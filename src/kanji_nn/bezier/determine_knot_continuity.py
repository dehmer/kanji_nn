import numpy as np
from svg.path import Path, CubicBezier


def determine_knot_continuity(path: Path, tol: float = 1e-5) -> list:
    """
    Determines the maximum C(n) continuity (up to C2) at each knot of a
    spline composed of CubicBezier segments from the svg.path library.

    Parameters:
        path (Path): An svg.path.Path object containing segments.
        tol (float): Tolerance for floating-point comparisons.

    Returns:
        list: A list of strings ('No C0', 'C0', 'C1', 'C2') for each knot
              between adjacent CubicBezier segments.
    """
    continuities = []

    # Filter to only look at CubicBezier segments for this calculation
    # (or ensure your path consists purely of them)
    segments = [seg for seg in path if isinstance(seg, CubicBezier)]

    if len(segments) < 2:
        return continuities  # Need at least two segments to have an internal knot

    for i in range(len(segments) - 1):
        seg1 = segments[i]
        seg2 = segments[i+1]

        # 1. Check C0 Continuity (Position)
        # End of segment 1 must equal start of segment 2
        p_end = np.array([seg1.end.real, seg1.end.imag])
        p_start = np.array([seg2.start.real, seg2.start.imag])

        if np.linalg.norm(p_end - p_start) > tol:
            continuities.append("No C0")
            continue  # If it's not C0, it cannot be C1 or C2

        # 2. Check C1 Continuity (First Derivative / Tangent)
        # For Cubic Bézier, derivative at t=1 is 3 * (P3 - P2)
        # Derivative at t=0 is 3 * (Q1 - Q0)
        p2 = np.array([seg1.control2.real, seg1.control2.imag])
        p3 = np.array([seg1.end.real, seg1.end.imag])

        q0 = np.array([seg2.start.real, seg2.start.imag])
        q1 = np.array([seg2.control1.real, seg2.control1.imag])

        d1_end = 3 * (p3 - p2)
        d1_start = 3 * (q1 - q0)
        if np.linalg.norm(d1_end - d1_start) > tol:
            continuities.append("C0")
            continue  # If it's not C1, it cannot be C2

        # 3. Check C2 Continuity (Second Derivative / Curvature)
        # For Cubic Bézier, second derivative at t=1 is 6 * (P1 - 2*P2 + P3)
        # Second derivative at t=0 is 6 * (Q0 - 2*Q1 + Q2)
        p1 = np.array([seg1.control1.real, seg1.control1.imag])
        q2 = np.array([seg2.control2.real, seg2.control2.imag])

        d2_end = 6 * (p1 - 2 * p2 + p3)
        d2_start = 6 * (q0 - 2 * q1 + q2)

        if np.linalg.norm(d2_end - d2_start) > tol:
            continuities.append("C1")
        else:
            continuities.append("C2")

    return continuities
