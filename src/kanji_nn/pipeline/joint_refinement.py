import numpy as np
from scipy.optimize import least_squares
from kanji_nn.svg.calculate_knot_angles import calculate_knot_angles
from svg.path import Path, Move, CubicBezier


def compute_u(pts):
    """Chord-length parametrization, matching fit_bezier_segment."""
    d = np.abs(np.diff(pts))
    s = np.concatenate(([0.0], np.cumsum(d)))
    total = s[-1]
    return np.zeros(len(pts)) if total < 1e-9 else s / total


def refine_knot(pts_i, u_i, P0, P1, pts_ip1, u_ip1, Q2, Q3,
                 P2_init, knot_init, Q1_init,
                 lambda_smooth=1.0, lambda_reg=50.0):
    """
    Jointly refine P2 (seg i, near handle), knot (P3 = Q0),
    and Q1 (seg i+1, near handle). P0, P1, Q2, Q3 stay fixed
    at whatever Part A already solved.
    """

    def unpack(x):
        return x[0] + 1j*x[1], x[2] + 1j*x[3], x[4] + 1j*x[5]

    def residuals(x):
        P2, knot, Q1 = unpack(x)

        B0, B1 = (1 - u_i)**3, 3*u_i*(1 - u_i)**2
        B2, B3 = 3*u_i**2*(1 - u_i), u_i**3
        res_i = (B0*P0 + B1*P1 + B2*P2 + B3*knot) - pts_i

        C0, C1 = (1 - u_ip1)**3, 3*u_ip1*(1 - u_ip1)**2
        C2, C3 = 3*u_ip1**2*(1 - u_ip1), u_ip1**3
        res_ip1 = (C0*knot + C1*Q1 + C2*Q2 + C3*Q3) - pts_ip1

        t_out, t_in = knot - P2, Q1 - knot
        n_out, n_in = abs(t_out), abs(t_in)
        cos_theta = ((t_out.real*t_in.real + t_out.imag*t_in.imag) / (n_out*n_in)
                     if n_out > 1e-9 and n_in > 1e-9 else 1.0)
        smooth_res = np.sqrt(lambda_smooth) * (1 - cos_theta)

        disp = knot - knot_init
        reg_res = np.sqrt(lambda_reg) * np.array([disp.real, disp.imag])

        return np.concatenate([res_i.real, res_i.imag,
                                res_ip1.real, res_ip1.imag,
                                [smooth_res], reg_res])

    x0 = np.array([P2_init.real, P2_init.imag,
                    knot_init.real, knot_init.imag,
                    Q1_init.real, Q1_init.imag])
    result = least_squares(residuals, x0)
    return unpack(result.x)


def joint_refinement(stroke):
    path = stroke.props["fitted"]
    angles = calculate_knot_angles(path)
    angle_threshold = 50.0

    if not any(a > angle_threshold for a in angles):
        return stroke

    seg_pts = stroke.props["segments"]
    segments = [seg for seg in path if isinstance(seg, CubicBezier)]
    assert len(segments) == len(seg_pts), "segment/point count mismatch"

    us = [compute_u(pts) for pts in seg_pts]

    for i, angle in enumerate(angles):
        if angle <= angle_threshold:
            continue  # leave well-behaved knots untouched

        seg_i, seg_ip1 = segments[i], segments[i + 1]

        P2, knot, Q1 = refine_knot(
            pts_i=seg_pts[i], 
            u_i=us[i],
            P0=seg_i.start, 
            P1=seg_i.control1,
            pts_ip1=seg_pts[i + 1], 
            u_ip1=us[i + 1],
            Q2=seg_ip1.control2, 
            Q3=seg_ip1.end,
            P2_init=seg_i.control2,
            knot_init=seg_i.end,
            Q1_init=seg_ip1.control1,
        )

        segments[i] = CubicBezier(seg_i.start, seg_i.control1, P2, knot)
        segments[i + 1] = CubicBezier(knot, Q1, seg_ip1.control2, seg_ip1.end)

    refined_path = Path(Move(segments[0].start), *segments)
    return stroke.clone(props={"fitted": refined_path}, force=True)
