import numpy as np
from svg.path import Path, CubicBezier
from .bezier import asarray, tangent_p1p3, tangent_p2p3, tangent_p0p1, tangent_p0p2


def calculate_knot_angles(path: Path) -> list:
    """
    Calculates the angular deviation (in degrees) at each knot.
    0 degrees means perfectly smooth (G1 continuous).
    Larger angles mean a sharper visual corner.

    Returns:
        list: A list of floats representing the turn angle at each knot.
    """
    angles = []
    segments = [seg for seg in path if isinstance(seg, CubicBezier)]

    if len(segments) < 2:
        return angles

    for i in range(len(segments) - 1):
        seg1 = asarray(segments[i])
        seg2 = asarray(segments[i+1])

        # Calculate tangent vectors at the knot
        v_in, norm_in = tangent_p2p3(seg1)
        v_out, norm_out = tangent_p0p1(seg2)

        epsilon = 1e-5

        # If a handle overlaps a knot (irregularity), we use the next best points
        if norm_in < epsilon:
            v_in, norm_in = tangent_p1p3(seg1)
        if norm_out < epsilon:
            v_out, norm_out = tangent_p0p2(seg2)

        # If vectors are still zero, the segment is collapsed
        if norm_in < epsilon or norm_out < epsilon:
            angles.append(0.0) # Assume smooth if undefined
            continue

        # Normalize the vectors
        v_in_u = v_in / norm_in
        v_out_u = v_out / norm_out

        # Calculate the angle between the two vectors using the dot product
        dot_product = np.clip(np.dot(v_in_u, v_out_u), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = float(np.degrees(angle_rad))

        angles.append(round(angle_deg, 2))

    return angles
