from dataclasses import dataclass, field, replace
import numpy as np

@dataclass(frozen=True)
class Stroke:
    dataset: str
    stroke_index: int
    raw: np.ndarray # [t, x, y, pressure]
    code_point: str
    literal: str
    stroke_type: tuple[str, str, str]
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

    @property
    def isbare(self): return len(self.props) == 0 and len(self.features) == 0

    def trim(self, region):
        return replace(
            self,
            raw = self.raw[region[0]:region[1], :],
            props = dict(),
            features = dict()
        )

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
