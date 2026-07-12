#!/usr/bin/env python3

import os
import re
from functools import partial
from dataclasses import dataclass, field, replace
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from kanji_nn.plot import multi_channel_plot, strokes_plot
from kanji_nn.conditioning import split_strokes
from kanji_nn.data import compose

identity = lambda x: x

@dataclass(frozen=True)
class Stroke:
    stroke_index: int
    raw: np.ndarray # [t, x, y, pressure]
    code_point: str
    literal: str
    features: dict[str, np.ndarray] = field(default_factory=dict)
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.raw.shape[0]

    @property
    def t(self): return self.raw[:, 0]

    @property
    def duration(self): return self.t[-1] - self.t[0]

    @property
    def start(self): return self.xy[0]

    @property
    def end(self): return self.xy[-1]

    @property
    def x(self): return self.raw[:, 1]

    @property
    def y(self): return self.raw[:, 2]

    @property
    def xy(self): return self.raw[:, (1, 2)]

    @property
    def pressure(self): return self.raw[:, 3]

    def clone(self, features = None, props = None, force = False):
        # Default to empty dictionaries if None is passed
        features = features or {}
        props = props or {}

        # Check for duplicate feature keys
        if not force:
            duplicate_fkeys = set(self.features) & set(features)
            if duplicate_fkeys:
                raise ValueError(f"[Stroke.clone()] duplicate feature key(s): {duplicate_fkeys}")

            duplicate_pkeys = set(self.props) & set(props)
            if duplicate_pkeys:
                raise ValueError(f"[Stroke.clone()] duplicate property key(s): {duplicate_pkeys}")

        # Verify feature shape alignment
        expected_rows = self.n_points
        for key, arr in features.items():
            # Ensure it is a numpy array
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"[Stroke.clone()] Feature '{key}' must be a numpy ndarray.")

            # Ensure row count (n) matches n_point
            if arr.shape[0] != expected_rows:
                raise ValueError(
                    f"[Stroke.clone()] Feature '{key}' row mismatch. "
                    f"Expected {expected_rows} rows, got {arr.shape[0]}."
                )

        # Merge and return the new object
        return replace(
            self,
            features=self.features | features,
            props=self.props | props
        )


def extract_code_point(filename):
    match = re.search(r"(U\+[0-9A-F]{4,5})", filename)
    if not match:
        raise ValueError(f'cp: invalid format in {filename}')
    return match.group(1)


class Character:
    def __init__(self, code_point, raw):
        """
        Fixed five column layout for raw (for now):
        0: timestamp (t)
        1: x coordinates
        2: y coordinates
        3: pressure
        4: pen-down/-up
        """
        self.code_point = code_point
        self.raw = raw
        self.literal = chr(int(code_point[2:], 16))

    @classmethod
    def of_npy(cls, filename):
        code_point = extract_code_point(filename)
        raw = np.load(filename)

        # Silently drop orientation and tile:
        if raw.shape[1] == 7:
            raw = raw[:, (0, 1, 2, 3, 6)]

        return cls(code_point, raw)

    # Stick either to raw xy or smoothened xy (not both).
    def strokes(self, smooth_fn = identity):
        strokes = split_strokes(self.raw)

        def smooth(raw):
            xy = smooth_fn(raw[:, 1:3])
            return np.column_stack((raw[:, 0], xy, raw[:, 3:]))

        return [Stroke(i, smooth(raw), self.code_point, self.literal) for i, raw in enumerate(strokes)]

# def metric_arc_length(stroke):
#     """
#     Arc length
#     ds: magnitude of vector [dxy(n), dxy(n+1)] = arc length between points n and n + 1
#     s_norm: normalized cumulative arc lengths [0, 1]
#     """

#     ds = np.linalg.norm(np.diff(stroke.xy, axis=0), axis=1)
#     ds = np.concatenate(([0.0], ds))
#     s = np.cumsum(ds)

#     # Avoid duplicate arc lengths: s += [0 * 1e-12, 1 * 1e-12, ..., (n-1) * 1 * 1e-12]
#     s += np.arange(len(s)) * 1e-12
#     s_norm = s / s[-1]

#     return stroke.clone(features={
#         "s": s,
#         "s_norm": s_norm
#     })

# def metric_tangent(stroke):
#     """
#     arc-length-based; feeds curvature
#     """
#     tx = np.gradient(stroke.x, stroke.features["s"])
#     ty = np.gradient(stroke.y, stroke.features["s"])
#     return stroke.clone(features={"tx": tx, "ty": ty})

# def metric_straightness(stroke):
#     """
#     Since s is the cumulative arc length of x,y itself, dx/ds and dy/ds are — by definition —
#     the components of the unit tangent vector. hypot(dx/ds, dy/ds) is mathematically
#     guaranteed to be ≈1 everywhere, for any curve, regardless of how fast or slow the hand
#     actually moved. I confirmed this on real data: mean=0.992, std=0.029, range=[0.828, 1.0].
#     That's not writing speed — it's just measuring how close your discretized tangent
#     vector is to unit length (which is a nice numerical sanity check, incidentally, but not
#     what the channel name promises).

#     Ratio of the direct shortcut to the actual two-hop path through each point.
#     => local straightness score, bounded to (0, 1]
#     """

#     s = stroke.features["s"]
#     dx = np.gradient(stroke.x, s)
#     dy = np.gradient(stroke.y, s)
#     straightness = np.hypot(dx, dy)

#     return stroke.clone(features={
#         "straightness": straightness
#     })


# def metric_speed(stroke):
#     # time-based; genuine writing speed
#     dx_dt = np.gradient(stroke.x, stroke.t)
#     dy_dt = np.gradient(stroke.y, stroke.t)
#     return stroke.clone(features={"speed": np.hypot(dx_dt, dy_dt)})


# def metric_curvature(stroke):
#     """
#     # Tangent angle
#     # theta:          θ (theta); tangent angle
#     # theta_gradient: change rate of θ in respect to s

#     # Curvature
#     # curvature: signed curvature dT/ds
#     # curvature_abs: absolute curvature dT/ds
#     """
#     tx = stroke.features["tx"]
#     ty = stroke.features["ty"]
#     s = stroke.features["s"]

#     theta = np.unwrap(np.arctan2(ty, tx))
#     theta_gradient = np.gradient(theta, s)

#     dtx = np.gradient(tx, s)
#     dty = np.gradient(ty, s)
#     curvature = tx * dty - ty * dtx

#     return stroke.clone(features={"θ": theta, "dθ/ds": theta_gradient, "K": curvature})


# def local_curvature(stroke, signed=True):
#     """
#     Per-sample turning angle (radians) between the incoming and outgoing
#     segment: heading(P_i -> P_i+1) minus heading(P_i-1 -> P_i), wrapped
#     to [-pi, pi]. Open-stroke convention: no wraparound, since assuming
#     the last point connects back to the first is a "this is a closed
#     loop" decision that belongs to Step 2, not here.

#     Turning angle rather than Menger/arc-length curvature deliberately:
#     dividing by segment length blows up whenever two samples are very
#     close together, which is exactly what happens at a slow-moving hook
#     apex -- the one place we most want a stable reading.

#     First and last samples have no turning angle defined -> NaN (not 0;
#     0 would mean "confirmed straight", which is a different claim).
#     """

#     x = stroke.x
#     y = stroke.y
#     n = len(x)

#     curvature = np.full(n, np.nan)
#     if n < 3:
#         return curvature

#     heading = np.arctan2(np.diff(y), np.diff(x))
#     dtheta = np.diff(heading)
#     dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

#     curvature[1:-1] = dtheta
#     if not signed:
#         curvature = np.abs(curvature)

#     return stroke.clone(features={"curvature": curvature})

def kinematics(stroke: Stroke) -> Stroke:
    t = stroke.t
    xy = stroke.xy

    # First-order differences, aligned with the current sample.
    dt = np.diff(t, prepend=np.nan)
    dxy = np.diff(xy, axis=0, prepend=np.full((1, 2), np.nan))

    dx = dxy[:, 0]
    dy = dxy[:, 1]

    ds = np.hypot(dx, dy)

    with np.errstate(divide='ignore', invalid='ignore'):
        velocity = dxy / dt[:, None]
        speed = ds / dt

    return stroke.clone(features={
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "ds": ds,
        "velocity": velocity,
        "speed": speed,
    })

def kinematics_2(stroke):
    dt = np.concatenate(([0.0], np.diff(stroke.t)))
    dx = np.concatenate(([0.0], np.diff(stroke.x)))
    dy = np.concatenate(([0.0], np.diff(stroke.y)))

    distance = np.sqrt(dx**2 + dy**2)
    vector = np.column_stack((dx, dy))
    magnitude = np.linalg.norm(vector, axis=1, keepdims=True)
    magnitude = np.where(magnitude == 0, 1.0, magnitude)
    unit_vector = vector / magnitude

    dot_product = np.sum(unit_vector[:-1] * unit_vector[1:], axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    velocity = distance / dt * 1000

    # θ (theta)
    angle = np.degrees(np.arccos(dot_product))
    angle = np.pad(angle, (0, 1), mode='constant', constant_values=np.nan)

    return stroke.clone(features={'velocity_2': velocity, 'angle_2': angle})


def process_file(filename):
    SIGMA = 1.0
    smooth_fn = lambda xy: gaussian_filter1d(xy, SIGMA, axis=0, mode='nearest') # reflect, nearest, mirror

    metrics_composed = compose(
        # partial(local_curvature, signed=False),
        # metric_curvature,
        # metric_tangent,
        # metric_speed,
        # metric_straightness,
        # metric_arc_length
        kinematics_2,
        kinematics
    )

    char = Character.of_npy(filename)
    strokes = char.strokes(smooth_fn=identity)
    for stroke in strokes:

        # add pressure as explicit feature for plot:
        stroke = stroke.clone(features={"pressure": stroke.pressure})
        stroke = metrics_composed(stroke)

        channels = ["pressure", "straightness", "θ", "dθ/ds", "K", "curvature"]
        channels = ["pressure", "velocity", "speed", "velocity_2", "angle_2"]
        multi_channel_plot(stroke, channels, figsize=(18, 8))
        # plt.savefig("kinematics")
        plt.show()


if __name__ == "__main__":
    dataset = 'katakana_49'
    dataset = 'hiragana-48'
    in_dir = f'data/dataset/{dataset}/npy-raw'

    def literal_to_hex(literal):
        return f'{ord(literal):x}'.upper()

    def infer_file_names(literals):
        return [f'U+{literal_to_hex(literal)}.npy' for literal in literals]

    white_list = []
    # white_list = infer_file_names(['ナ'])

    for (dirpath, dirnames, filenames) in os.walk(in_dir):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if len(white_list) and not filename in white_list: continue
            process_file(f'{dirpath}/{filename}')
