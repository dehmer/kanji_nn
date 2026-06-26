#!/usr/bin/env python3

import numpy as np
import kanji_nn.plot as plot



raw = np.load('strokes/U+306A.npy')
plot.plot_dynamics(raw)

# split_indices = np.where(raw[:, 2] == 0)[0] + 1
# strokes = np.split(raw, split_indices[:-1])

# for stroke in strokes:
#     print('#points:', len(stroke)) # n point
#     timestamps = stroke[:,3]
#     timestamp_diffs = np.diff(timestamps)

#     # Normalized sum of timestamp differences;
#     # starting with (close to) 0.
#     elapsed = np.cumsum(timestamp_diffs)
#     xs = stroke[:,1]
#     ys = stroke[:,2]
#     x_diffs = xs[1:] - xs[:-1] # n - 1
#     y_diffs = ys[1:] - ys[:-1] # n - 1

#     # euclidian distances of consecutive points: n - 1
#     dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)

#     velocities = dists / timestamp_diffs * 1000
#     print(velocities)