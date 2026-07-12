Topic: Immutable data class
Bot: ChatGPT
Date: 26 07 11

> Say I have a 2D NumPy array of vector stroke/line data with n point a columns [timestamp, x, y pressure and pen-down/-up state]. Now I want to calculate a couple of kinematic/geometric metrics on a per point basis. I'm struggling to come up with function interfaces for the different calculations which are practical and convenient. In general I want to keep intermediate results from one calculation which might be re-used by a downstream calculation. I'm going back and forth between different approaches. Currently I'm thinking about incrementally (NumPy) hstacking new columns for each calculation and maintaining a dictionary with column specs: `data_name -> index/indices`. Each function would roughly look like `fn(cols: dict[str, int], data: npt.NDArray[np.float32]) -> tuple[dict[str, int], npt.NDArray[np.float32]]. What do you think?

Your instinct to preserve intermediate results is good, but I would avoid encoding your pipeline as "a NumPy array that keeps growing new columns." It works initially, but tends to become awkward over time:

* Every function needs to know column indices.
* Column bookkeeping becomes increasingly complex.
* `np.hstack` reallocates the entire array every time (unless you batch additions).
* Functions become less self-documenting (`cols["speed"]` vs `speed`).

Instead, I'd separate **geometry storage** from **derived features**.

## Option 1: Feature dictionary (my preferred approach)

Keep the original data immutable, and store derived quantities separately.

```python
@dataclass
class StrokeData:
    raw: np.ndarray              # (n, 5)
    features: dict[str, np.ndarray] = field(default_factory=dict)
```

Now calculations become

```python
def velocity(data: StrokeData) -> None:
    t = data.raw[:, 0]
    x = data.raw[:, 1]
    y = data.raw[:, 2]

    dt = np.diff(t, prepend=t[0])
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    speed = np.sqrt(dx**2 + dy**2) / np.maximum(dt, 1e-6)

    data.features["dt"] = dt
    data.features["speed"] = speed
```

Then

```python
def curvature(data: StrokeData) -> None:
    if "speed" not in data.features:
        velocity(data)

    speed = data.features["speed"]
    ...
```

Advantages:

* No column bookkeeping.
* Named arrays instead of indices.
* Easy dependency management.
* Original data never changes.

---

## Option 2: Small typed class

If the "raw" columns are fixed forever, I'd expose them as properties.

```python
@dataclass
class Stroke:
    data: np.ndarray
    features: dict[str, np.ndarray]

    @property
    def t(self):
        return self.data[:, 0]

    @property
    def x(self):
        return self.data[:, 1]

    @property
    def y(self):
        return self.data[:, 2]

    @property
    def pressure(self):
        return self.data[:, 3]
```

Now your code becomes

```python
dx = np.diff(stroke.x)
```

instead of

```python
dx = np.diff(data[:, cols["x"]])
```

which is much clearer.

---

## Option 3: Dependency graph

If you're computing dozens of metrics, each depending on others:

```
speed
  ↑
velocity
  ↑
acceleration
  ↑
jerk
```

you can centralize feature computation.

```python
FEATURES = {
    "velocity": compute_velocity,
    "speed": compute_speed,
    "curvature": compute_curvature,
}
```

and

```python
stroke.require("curvature")
```

would recursively compute missing prerequisites only once.

This scales surprisingly well.

---

## Option 4: Structured arrays

People often suggest structured NumPy arrays:

```python
dtype = [
    ("t", "f4"),
    ("x", "f4"),
    ("y", "f4"),
    ("pressure", "f4"),
    ("state", "u1"),
]
```

Personally, I wouldn't.

They look attractive, but they're poorly supported by much of the NumPy ecosystem and are inconvenient for vectorized linear algebra.

---

## I would avoid growing the array

Repeatedly doing

```python
data = np.hstack([data, new_column])
```

has a few drawbacks:

* every `hstack` copies the entire array;
* column indices become an API that all functions must understand;
* adding multi-column outputs (e.g. velocity as `(vx, vy)`) complicates the index mapping;
* downstream code has to remember whether `"velocity"` refers to one column or two.

---

## If you really like "everything in one array"

At least preallocate.

```python
data = np.empty((n, 32), dtype=np.float32)

cols = {
    "t": 0,
    "x": 1,
    "y": 2,
    "pressure": 3,
    "state": 4,
}
```

Then each feature claims unused columns instead of calling `hstack`.

---

## My recommendation

Given that you're building a pipeline of kinematic and geometric computations, I'd use something like:

```python
@dataclass
class Stroke:
    raw: np.ndarray                      # original data
    features: dict[str, np.ndarray]      # computed per-point arrays
```

with computation functions like

```python
def compute_speed(stroke: Stroke) -> np.ndarray:
    ...
    stroke.features["speed"] = speed
    return speed
```

or even

```python
speed = stroke.get("speed")
curvature = stroke.get("curvature")
```

where `get()` computes missing features lazily.

This keeps the raw observations separate from derived quantities, makes dependencies explicit, and avoids tying every function to a shared column-index convention. It also scales naturally from a handful of metrics to dozens without making the array layout itself part of your API.

> Good advice! I will check it out.

I think you'll find it stays manageable much longer.

One additional pattern that's worked well for scientific/analysis code is to distinguish between **primitive** and **derived** features.

For example:

```python
stroke.raw
├── t
├── x
├── y
├── pressure
└── pen_state
```

Derived features might then look like:

```python
stroke.features
├── dt
├── dx
├── dy
├── distance
├── speed
├── acceleration
├── tangent
├── normal
├── curvature
└── ...
```

Then each computation simply declares what it needs:

```python
def compute_speed(stroke: Stroke):
    dt = stroke.feature("dt")
    distance = stroke.feature("distance")
    stroke.features["speed"] = distance / dt
```

where `stroke.feature("dt")` computes `dt` on demand if it isn't already cached. That way:

* each feature is computed at most once,
* computation order no longer matters,
* dependencies stay local to the functions that use them.

This is essentially a lightweight version of a directed acyclic graph (DAG), but without introducing a framework.

One other suggestion: don't hesitate to store features with different shapes. For example,

```python
features = {
    "speed":      (n,),
    "velocity":   (n, 2),
    "tangent":    (n, 2),
    "bbox":       (4,),
    "path_len":   (),      # scalar
}
```

A dictionary of arrays has no requirement that everything be a single `(n, m)` matrix, which gives you much more flexibility than continually adding columns.

Finally, if this grows into a reusable library, you could wrap the cache behind a small API:

```python
stroke["speed"]        # cached feature
stroke.compute("speed")
stroke.invalidate()    # after modifying raw data
```

This hides the implementation details, so you can later change how features are cached or computed without changing the rest of your code.

Overall, I think separating **raw data** from **derived features** is a cleaner abstraction than treating everything as columns in one expanding array. It aligns well with how scientific computing libraries tend to evolve and makes it much easier to add new metrics later.

> Can we make `Stroke` immutable and add a copy constructor to create a shallow copy and add additional feature data?

Yes—and I actually like that design even better. It fits nicely with a functional pipeline where each computation returns a new `Stroke` that shares all existing data but adds (or replaces) features.

Because NumPy arrays are already reference types, a shallow copy is cheap: you're only creating a new dictionary and dataclass instance, not copying the arrays themselves.

For example:

```python
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class Stroke:
    raw: npt.NDArray[np.float32]
    features: Mapping[str, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def with_features(self, **new_features: np.ndarray) -> "Stroke":
        features = dict(self.features)
        features.update(new_features)
        return Stroke(
            raw=self.raw,
            features=MappingProxyType(features),
        )
```

Then your functions become pure:

```python
def compute_speed(stroke: Stroke) -> Stroke:
    ...
    return stroke.with_features(speed=speed, dt=dt)
```

and chaining is straightforward:

```python
stroke = Stroke(raw)
stroke = compute_velocity(stroke)
stroke = compute_speed(stroke)
stroke = compute_curvature(stroke)
```

## An even nicer API

Personally, I'd make `with_features()` accept a mapping instead of `**kwargs`:

```python
def with_features(self, features: Mapping[str, np.ndarray]) -> "Stroke":
    merged = dict(self.features)
    merged.update(features)
    return Stroke(
        self.raw,
        MappingProxyType(merged),
    )
```

Then a function can naturally return multiple results:

```python
return stroke.with_features({
    "dx": dx,
    "dy": dy,
    "distance": distance,
})
```

without worrying about keyword names.

## Caching vs. immutability

The only thing to think about is whether you want **lazy evaluation**. If you do, complete immutability becomes trickier because computing `"speed"` would need to return a *new* `Stroke`:

```python
stroke2 = stroke.require("speed")
```

rather than mutating an internal cache.

That's not a bad API at all—in fact it's quite common in persistent data structures—but it's a different style from mutable memoization.

## One caveat

A frozen dataclass doesn't make the underlying arrays immutable:

```python
stroke.raw[0, 0] = 42   # still works
```

because `frozen=True` only prevents rebinding the `raw` attribute.

If you truly want `Stroke` to be immutable, you can mark the arrays as read-only:

```python
raw = raw.copy()
raw.flags.writeable = False
```

or

```python
raw.flags.writeable = False
```

if you know nobody else should modify them.

The same applies to feature arrays before storing them.

---

Overall, I think this immutable design is a very good fit for your use case. It gives you referential transparency (a `Stroke` never changes after creation), makes testing easier, and allows different processing pipelines to branch from the same input without worrying about accidental mutation. The copying cost is minimal because the large NumPy arrays are shared; only the feature mapping itself is recreated.
