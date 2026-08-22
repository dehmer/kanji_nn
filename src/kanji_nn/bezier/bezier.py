from typing import Literal, Tuple
from svg.path import CubicBezier
import numpy as np
import numpy.typing as npt


# evaluates cubic bezier at t, return point
def q(ctrlPoly, t):
    return (1.0-t)**3 * ctrlPoly[0] + 3*(1.0-t)**2 * t * ctrlPoly[1] + 3*(1.0-t)* t**2 * ctrlPoly[2] + t**3 * ctrlPoly[3]


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    return 3*(1.0-t)**2 * (ctrlPoly[1]-ctrlPoly[0]) + 6*(1.0-t) * t * (ctrlPoly[2]-ctrlPoly[1]) + 3*t**2 * (ctrlPoly[3]-ctrlPoly[2])


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    return 6*(1.0-t) * (ctrlPoly[2]-2*ctrlPoly[1]+ctrlPoly[0]) + 6*(t) * (ctrlPoly[3]-2*ctrlPoly[2]+ctrlPoly[1])


CubicBezierMatrix = npt.NDArray[Literal[4], Literal[2], np.float64]


def asarray(s: CubicBezier) -> CubicBezierMatrix:
    """Converts a CubicBezier curve segment into a (4, 2) 2D float array."""
    return np.array([
        [p.real, p.imag] for p in (s.start, s.control1, s.control2, s.end)
    ])


def tangent(
    s: CubicBezier | CubicBezierMatrix,
    start: int,
    end: int
) -> Tuple[npt.NDArray[np.float64], float]:
    if isinstance(s, CubicBezier):
        return tangent(asarray(s), start, end)

    v = s[end, :] - s[start, :]
    return v, np.linalg.norm(v)


def tangent_p2p3(s: CubicBezier):
    return tangent(s, 2, 3)


def tangent_p1p3(s: CubicBezier):
    return tangent(s, 1, 3)


def tangent_p0p1(s: CubicBezier):
    return tangent(s, 0, 1)


def tangent_p0p2(s: CubicBezier):
    return tangent(s, 0, 2)
