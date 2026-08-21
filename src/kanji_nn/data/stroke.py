from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np

@dataclass(frozen=True, kw_only=True)
class Stroke:
    dataset: str

    # literal/stroke_index/stroke_count
    key: str
    features: dict[str, np.ndarray] = field(default_factory=dict)
    props: dict[str, Any] = field(default_factory=dict)
    sticky: dict[str, Any] = field(default_factory=dict)

    @property
    def literal(self) -> str:
        literal, _, _ = self.key.split("/")
        return literal

    @property
    def stroke_index(self) -> int:
        _, index, _ = self.key.split("/")
        return int(index)

    @property
    def stroke_count(self) -> int:
        """
        Overall character stroke count.
        Useful if we want to perform operations on a complete
        character in the pipeline.
        """
        _, _, count = self.key.split("/")
        return int(count)

    @property
    def code_point(self) -> str:
        return f"U+{ord(self.literal[0]):04X}"

    def clone(self, features=None, props=None, sticky=None, force=False):
        # Default to empty dictionaries if None is passed
        features = features or {}
        props = props or {}
        sticky = sticky or {}

        # Check for duplicate feature keys
        if not force:
            duplicate_fkeys = set(self.features) & set(features)
            if duplicate_fkeys:
                raise ValueError(f"[Stroke.clone()] duplicate feature key(s): {duplicate_fkeys}")

            duplicate_pkeys = set(self.props) & set(props)
            if duplicate_pkeys:
                raise ValueError(f"[Stroke.clone()] duplicate property key(s): {duplicate_pkeys}")

        # Verify feature shape alignment

        lengths = set([len(arr) for arr in (self.features | features).values()])
        if len(lengths) != 1:
            raise TypeError(f"[Stroke.clone()] Feature row count mismatch.")

        # Merge and return the new object
        return replace(
            self,
            features=self.features | features,
            props=self.props | props,
            sticky=self.sticky | sticky,
        )
