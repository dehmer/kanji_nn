import numpy as np

def bezier_handle_geometry(segment):
    p0 = np.array([segment.start.real, segment.start.imag])
    p1 = np.array([segment.control1.real, segment.control1.imag])
    p2 = np.array([segment.control2.real, segment.control2.imag])
    p3 = np.array([segment.end.real, segment.end.imag])

    chord = p3 - p0
    chord_length = np.linalg.norm(chord)

    h1 = p1 - p0
    h2 = p2 - p3

    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    signed_angle = lambda v, ref: np.arctan2(cross(ref, v), np.dot(ref, v))

    return {
        "chord_length": chord_length,
        "directions": (signed_angle(h1, chord), signed_angle(h2, chord)),
        "magnitudes": (np.linalg.norm(h1) / chord_length, np.linalg.norm(h2) / chord_length),
    }
