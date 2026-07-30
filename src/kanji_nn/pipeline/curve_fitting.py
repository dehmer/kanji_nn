import numpy as np
from svg.path import CubicBezier, Move, Path
from kanji_nn.svg.bezier_obb import bezier_obb
from kanji_nn.svg.classify_bezier import classify_bezier


def rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])


def similarity_transform(vs_geom):
    vs, geom = vs_geom
    mag1, mag2 = geom["magnitudes"]
    angle1, angle2 = geom["directions"]
    v_chord = vs[-1] - vs[0]
    v_chord_length = np.linalg.norm(v_chord)
    h1_v = rotate(v_chord, angle1) * mag1
    h2_v = rotate(v_chord, angle2) * mag2
    p1_seed = vs[0] + h1_v
    p2_seed = vs[-1] + h2_v
    return (p1_seed, p2_seed)


def fit_cubic(vs_seeds):
    vertices, seed = vs_seeds
    p0, p3 = vertices[0], vertices[-1]
    p1_seed, p2_seed = seed

    t1_hat = (p1_seed - p0) / np.linalg.norm(p1_seed - p0)
    t2_hat = (p2_seed - p3) / np.linalg.norm(p2_seed - p3)

    # chord-length parameterization of vertices, t in [0, 1]
    d = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))))
    u = d / d[-1]

    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u ** 2 * (1 - u)
    b3 = u ** 3

    a1 = b1[:, None] * t1_hat   # (n, 2)
    a2 = b2[:, None] * t2_hat   # (n, 2)
    rhs = vertices - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3

    C = np.array([
        [np.sum(a1 * a1), np.sum(a1 * a2)],
        [np.sum(a1 * a2), np.sum(a2 * a2)],
    ])
    X = np.array([np.sum(a1 * rhs), np.sum(a2 * rhs)])

    alpha1, alpha2 = np.linalg.solve(C, X)

    control1 = p0 + alpha1 * t1_hat
    control2 = p3 + alpha2 * t2_hat

    return CubicBezier(
        complex(*p0), complex(*control1), complex(*control2), complex(*p3)
    )


def curve_fitting(stroke):
    segments = stroke.props["segments"]

    # handle geometries:
    # 1. get directions of (p1-p0), (p2-p3) relative to chord (p3-p0)
    # 2. get magnitudes, relative to chord length: (|p1-p0|/|p3-p0|), (|p2-p3|/|p3-p0|)
    #
    vs_geoms = [
        (x[0], bezier_handle_geometry(x[1]))
        for x in segments
        if isinstance(x[1], CubicBezier)
    ]

    # 3. (similarity) transform (rotate, scale) to match (v(n-1) - v(0))
    #
    vs_seeds = [(x[0], similarity_transform(x)) for x in vs_geoms]

    # 4. use p1, p2 seed to fit curve/vertices to cubic bezier
    #
    v0 = segments[0][0][0] # segment # 0 -> vertices -> vertices # 0
    move = Move(complex(*v0), relative=False)
    fitted_path = Path(*[move] + [fit_cubic(x) for x in vs_seeds])

    def quad(vs_seeds):
        vs, ps = vs_seeds
        return np.asarray([vs[0], ps[0], ps[1], vs[-1], vs[0]])

    quads = [quad(t) for t in vs_seeds]
    return stroke.clone(props={"quads": quads, "fitted_path": fitted_path})
