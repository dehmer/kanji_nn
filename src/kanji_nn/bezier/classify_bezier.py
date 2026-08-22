import math
from svg.path import CubicBezier

def classify_bezier(curve: CubicBezier, epsilon: float = 0.05) -> str:
    """Classifies a cubic Bezier curve into one of five categories.

    Categories: 'empty', 'near-straight', 'left-bend', 'right-bend', 's-bend'
    """
    # 1. Extract complex coordinates from svg.path CubicBezier object
    p0 = curve.start
    p1 = curve.control1
    p2 = curve.control2
    p3 = curve.end

    # 2. Compute baseline vector (B = p3 - p0)
    bx = p3.real - p0.real
    by = p3.imag - p0.imag
    b_len = math.hypot(bx, by)

    # 3. Handle Coincident Endpoints ('empty')
    if b_len < 1e-9:
        return 'empty'

    # 4. Calculate Signed Perpendicular Distances using 2D cross-product
    # Positive = Left side, Negative = Right side (standard Cartesian)
    d1 = ((p1.real - p0.real) * by - (p1.imag - p0.imag) * bx) / b_len
    d2 = ((p2.real - p0.real) * by - (p2.imag - p0.imag) * bx) / b_len

    # 5. Define Dynamic Near-Straight Threshold
    threshold = epsilon * b_len

    # 6. Classification Logic
    abs_d1, abs_d2 = abs(d1), abs(d2)

    if abs_d1 <= threshold and abs_d2 <= threshold:
        return 'near-straight'

    s1 = 0 if abs_d1 <= threshold else math.copysign(1, d1)
    s2 = 0 if abs_d2 <= threshold else math.copysign(1, d2)

    if s1 * s2 < 0:
        return 's-bend'

    # Positive cross-product in SVG space means a visual RIGHT turn.
    net_displacement = d1 + d2
    if net_displacement > 0:
        return 'right-bend'
    else:
        return 'left-bend'