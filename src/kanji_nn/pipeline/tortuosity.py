import numpy as np

def tortuosity(stroke, w=5):

    # arc-length between two consecutive vertices:
    ds = stroke.features["raw:ds"]
    xy = stroke.xy

    radius = w // 2
    min_idx = radius
    max_idx = len(xy) - 1 - radius

    # Define the start (S) and end (E) indices for all windows
    # centered at the inner vertices [min_idx, max_idx] [2]
    S = np.arange(min_idx - radius, max_idx - radius + 1)
    E = S + (2 * radius)

    # Calculate arc lengths for all windows using cumulative sum
    # ds[j] is the distance between points j-1 and j [1, 2]
    cumulative_ds = np.cumsum(ds)
    arc_lengths = cumulative_ds[E] - cumulative_ds[S]

    # Calculate chord lengths (Euclidean distance) for all windows simultaneously [2]
    chords = np.linalg.norm(xy[E] - xy[S], axis=1)

    # Calculate tortuosity values
    tortuosity = arc_lengths / chords

    tortuosity = np.concat([[1.0] * radius, tortuosity, [1.0] * radius])
    return stroke.clone(features={"tortuosity": tortuosity})
