import numpy as np
from svg.path import Path, CubicBezier


def is_regular_path(path: Path, tol: float = 1e-5) -> bool:
    """
    Checks if an svg.path.Path composed of CubicBezier segments is regular.
    A path is regular if its velocity vector is never zero.

    Parameters:
        path (Path): The parsed svg.path object.
        tol (float): Numerical tolerance for zero-velocity checks.

    Returns:
        bool: True if the path is regular, False if it has an irregularity.
    """
    segments = [seg for seg in path if isinstance(seg, CubicBezier)]
    if not segments:
        return True  # Empty path or no Bézier segments is technically safe

    for seg in segments:
        # Convert control points to 2D numpy arrays
        p0 = np.array([seg.start.real, seg.start.imag])
        p1 = np.array([seg.control1.real, seg.control1.imag])
        p2 = np.array([seg.control2.real, seg.control2.imag])
        p3 = np.array([seg.end.real, seg.end.imag])

        # 1. Check Knot Stagnation (Velocity at boundaries t=0 and t=1)
        # Velocity at t=0 is 3*(P1 - P0). If P1 == P0, velocity is zero.
        if np.linalg.norm(p1 - p0) < tol:
            return False
        # Velocity at t=1 is 3*(P3 - P2). If P3 == P2, velocity is zero.
        if np.linalg.norm(p3 - p2) < tol:
            return False

        # 2. Check Internal Cusps (Velocity is zero somewhere inside 0 < t < 1)
        # The derivative of a Cubic Bézier is a quadratic equation:
        # B'(t) = 3 * [ c * t^2 + b * t + a ]
        A = p1 - p0
        B = p2 - p1
        C = p3 - p2

        cx, bx, ax = (A[0] - 2*B[0] + C[0]), 2*(B[0] - A[0]), A[0]
        cy, by, ay = (A[1] - 2*B[1] + C[1]), 2*(B[1] - A[1]), A[1]

        # Find where the X-velocity drops to zero
        roots_x = []
        if abs(cx) < tol:
            if abs(bx) > tol:
                roots_x.append(-ax / bx)
        else:
            disc = bx**2 - 4*cx*ax
            if disc >= 0:
                roots_x.append((-bx + np.sqrt(disc)) / (2*cx))
                roots_x.append((-bx - np.sqrt(disc)) / (2*cx))

        # Filter roots to see if any occur strictly inside the curve segment
        valid_roots_x = [t for t in roots_x if tol < t < 1 - tol]

        # Check if the Y-velocity also drops to zero at that exact same time 't'
        for t in valid_roots_x:
            y_prime = 3 * (cy * (t**2) + by * t + ay)
            if abs(y_prime) < tol:
                return False  # Both X and Y velocities are 0. It's a cusp!

    return True
