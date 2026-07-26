import numpy as np
from svg.path import CubicBezier

def bezier_obb(segment: CubicBezier):
    """
    Calculates the Oriented Bounding Box (OBB) for an svg.path CubicBezier segment.

    Returns:
        dict: OBB properties including a 5x2 NumPy array for a closed polygon.
    """
    # 1. Extract control points
    p0 = segment.start
    p1 = segment.control1
    p2 = segment.control2
    p3 = segment.end

    # 2. Establish local coordinate baseline
    baseline = p3 - p0
    width_base = abs(baseline)

    # Handle edge case where start and end points are identical
    if width_base < 1e-9:
        pts = [p0, p1, p2, p3]
        min_x = min(p.real for p in pts)
        max_x = max(p.real for p in pts)
        min_y = min(p.imag for p in pts)
        max_y = max(p.imag for p in pts)

        w, h = max_x - min_x, max_y - min_y
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        return {
            'origin': corners[0],
            'direction': (1.0, 0.0),
            'width': w,
            'height': h,
            'corners': corners,
            'polygon': np.array(corners + [corners[0]]), # 5x2 closed path
            'ratio': None
        }

    # Normalized direction vector
    u_dir = baseline / width_base
    u_dir_conj = u_dir.conjugate()

    # 3. Transform points into local space (flattened to X-axis)
    local_p0 = 0.0 + 0.0j
    local_p1 = (p1 - p0) * u_dir_conj
    local_p2 = (p2 - p0) * u_dir_conj
    local_p3 = complex(width_base, 0.0)

    local_pts = [local_p0, local_p1, local_p2, local_p3]

    # 4. Compute local space bounds
    min_l_x = min(p.real for p in local_pts)
    max_l_x = max(p.real for p in local_pts)
    min_l_y = min(p.imag for p in local_pts)
    max_l_y = max(p.imag for p in local_pts)

    obb_width = max_l_x - min_l_x
    obb_height = max_l_y - min_l_y

    # 5. Define local corners
    local_corners = [
        complex(min_l_x, min_l_y),  # Bottom-Left
        complex(max_l_x, min_l_y),  # Bottom-Right
        complex(max_l_x, max_l_y),  # Top-Right
        complex(min_l_x, max_l_y)   # Top-Left
    ]

    # 6. Map back to global space and convert to (x, y) tuples
    global_corners_complex = [(pt * u_dir) + p0 for pt in local_corners]
    global_corners = [(p.real, p.imag) for p in global_corners_complex]

    # 7. Create closed 5-row loop (BL -> BR -> TR -> TL -> BL)
    closed_polygon = global_corners + [global_corners[0]]

    return {
        'origin': global_corners[0],
        'direction': (u_dir.real, u_dir.imag),
        'width': obb_width,
        'height': obb_height,
        'corners': global_corners,
        'polygon': np.array(closed_polygon), # Shape (5, 2)
        'ratio': obb_height / obb_width
    }
