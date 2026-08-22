from svg.path import (
    Path,
    Move,
    Line,
    QuadraticBezier,
    CubicBezier,
    Arc,
    Close,
)

def transform(path: Path, a: complex = 1, b: complex = 0) -> Path:
    """Return a transformed copy of an svg.path.Path.

    Applies the affine transform

        z' = a * z + b

    to every point in every segment.
    """

    def T(z: complex) -> complex:
        return a * z + b

    transformed = []

    for seg in path:
        if isinstance(seg, Move):
            transformed.append(
                Move(T(seg.start), T(seg.end))
            )

        elif isinstance(seg, Line):
            transformed.append(
                Line(T(seg.start), T(seg.end))
            )

        elif isinstance(seg, QuadraticBezier):
            transformed.append(
                QuadraticBezier(
                    T(seg.start),
                    T(seg.control),
                    T(seg.end),
                )
            )

        elif isinstance(seg, CubicBezier):
            transformed.append(
                CubicBezier(
                    T(seg.start),
                    T(seg.control1),
                    T(seg.control2),
                    T(seg.end),
                )
            )

        elif isinstance(seg, Arc):
            # Radius scales with |a|. Rotation handling assumes
            # a represents uniform scaling and rotation.
            transformed.append(
                Arc(
                    T(seg.start),
                    seg.radius * abs(a),
                    seg.rotation,
                    seg.arc,
                    seg.sweep,
                    T(seg.end),
                )
            )

        elif isinstance(seg, Close):
            transformed.append(
                Close(T(seg.start), T(seg.end))
            )

        else:
            raise TypeError(f"Unsupported segment type: {type(seg).__name__}")

    return Path(*transformed)