import numpy as np
from svg.path import Path, CubicBezier


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
        seg1 = segments[i]
        seg2 = segments[i+1]

        # Get the control points that dictate the tangents at the knot
        p2 = np.array([seg1.control2.real, seg1.control2.imag])
        p3 = np.array([seg1.end.real, seg1.end.imag])

        q0 = np.array([seg2.start.real, seg2.start.imag])
        q1 = np.array([seg2.control1.real, seg2.control1.imag])

        # Calculate tangent vectors at the knot
        v_in = p3 - p2     # Incoming direction vector
        v_out = q1 - q0    # Outgoing direction vector

        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        # If a handle overlaps a knot (irregularity), we use the next best points
        if norm_in < 1e-5:
            p1 = np.array([seg1.control1.real, seg1.control1.imag])
            v_in = p3 - p1
            norm_in = np.linalg.norm(v_in)
        if norm_out < 1e-5:
            q2 = np.array([seg2.control2.real, seg2.control2.imag])
            v_out = q2 - q0
            norm_out = np.linalg.norm(v_out)

        # If vectors are still zero, the segment is collapsed
        if norm_in < 1e-5 or norm_out < 1e-5:
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
