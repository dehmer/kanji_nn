#!/usr/bin/env python3

from os import walk
import numpy as np
import kanji_nn.plot as plot
import matplotlib.pyplot as plt

def split_into_strokes(raw):
    # drop pen status and split into strokes
    split_indices = np.where(raw[:, 4] == 0)[0] + 1
    raw = raw[:, :-1]
    return np.split(raw, split_indices[:-1])


def compose_from_strokes(strokes):
    chunks = []
    for stroke in strokes:
        # pen status: 111....110
        status = np.ones((len(stroke), 1), dtype=np.float32)
        status[-1] = 0

        # insert status
        chunk = np.insert(stroke, [2], status, axis=1)
        chunks.append(chunk)
    return np.vstack(chunks)

def normalize_array(arr):
    """Normalizes a numpy array to the range [0-1], safely handling NaNs."""
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def plot_dynamics(ts, velocities, angles):
    """Plots velocities and angles with a shared x-axis starting at 0.

    Fixes the tooltip value evaluation issue by tracking data directly via numpy
    indexing.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    # --- Plot Data ---
    color1 = "tab:blue"
    ax1.set_xlabel("Time (ts)", fontweight="bold")
    ax1.set_ylabel("Velocity", color=color1, fontweight="bold")
    line1 = ax1.plot(ts, velocities, color=color1, label="Velocities", lw=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    color2 = "tab:red"
    ax2.set_ylabel("Angle", color=color2, fontweight="bold")
    line2 = ax2.plot(
        ts, angles, color=color2, linestyle="--", label="Angles", lw=2
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # --- Enforce X-Axis Constraints ---
    ax1.set_xlim(left=0)

    # --- Combine Legends ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.title(
        "Velocity and Angle Tracking Over Time", fontweight="bold", fontsize=14
    )

    # --- Interactive Hover Elements ---
    v_line = ax1.axvline(color="gray", linestyle=":", alpha=0.8, visible=False)

    bbox_props = dict(
        boxstyle="round,pad=0.5", fc="yellow", alpha=0.8, ec="black", lw=1
    )

    # Use 'axes fraction' for xycoords so the tooltip position scales reliably
    # independent of fluctuating data coordinates between ax1 and ax2
    tooltip = ax1.annotate(
        "",
        xy=(0, 0),
        xycoords="data",
        xytext=(15, 15),
        textcoords="offset points",
        bbox=bbox_props,
        fontweight="bold",
    )
    tooltip.set_visible(False)

    def on_mouse_move(event):
        # Trigger if the mouse is in either the velocity or angle axis areas
        if event.inaxes not in [ax1, ax2]:
            if v_line.get_visible():
                v_line.set_visible(False)
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
            return

        mouse_x = event.xdata

        # Safely extract index from the main numpy timestamp array
        idx = np.searchsorted(ts, mouse_x)
        idx = np.clip(idx, 0, len(ts) - 1)

        if idx > 0 and abs(mouse_x - ts[idx - 1]) < abs(mouse_x - ts[idx]):
            idx -= 1

        # Extract precise data from the raw arrays directly
        current_ts = ts[idx]
        current_vel = velocities[idx]
        current_ang = angles[idx]

        # Update vertical cursor line
        v_line.set_xdata([current_ts, current_ts])
        v_line.set_visible(True)

        # Structure text string safely
        tooltip_text = (
            f"ts: {current_ts:.2f}\n"
            f"Vel: {current_vel:.2f}\n"
            f"Ang: {current_ang:.2f}"
        )
        tooltip.set_text(tooltip_text)

        # Place the tooltip at the current point's x-coordinate,
        # but safely aligned to the vertical center of ax1's viewport
        y_min, y_max = ax1.get_ylim()
        tooltip.xy = (current_ts, (y_min + y_max) / 2)
        tooltip.set_visible(True)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    fig.tight_layout()
    plt.show()

def telemetry(strokes):
    TIME_STAMP, DX, DY, PRESSURE, FEATURE = 0, 1, 2, 3, 4

    # xs, ys, ts = stroke[:, 0], stroke[:, 1], stroke[:, 2]
    diffs = np.diff(strokes, axis=0)
    dt, dx, dy = diffs[:, TIME_STAMP], diffs[:, DX], diffs[:, DY]

    # 2. Compute metrics
    distances = np.sqrt(dx**2 + dy**2)
    velocities = distances / dt

    vectors = np.column_stack((dx, dy))
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes = np.where(magnitudes == 0, 1.0, magnitudes)
    unit_vectors = vectors / magnitudes
    dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot_products))

    # padding
    velocities = velocities * 1000
    velocities = np.pad(velocities, (1, 0), mode='constant', constant_values=np.nan)
    angles = np.pad(angles, (1, 1), mode='constant', constant_values=np.nan)

    # low velocity, high angle
    stacked = np.vstack((velocities, angles)).T
    candidates = np.where((stacked[:, 0] < 10) & (stacked[:, 1] > 60))
    print('[candidates]', candidates)
    for index in candidates:
        print('ts', strokes[index, TIME_STAMP])

    plot_dynamics(strokes[:, TIME_STAMP], velocities, angles)


def process(filename):
    # [timestamp[0]:x[1]:y[2]:pressure[3]:feature[4]]
    raw = np.load(f'strokes/{filename}')
    telemetry(raw)
    # strokes = split_into_strokes(raw)
    # for stroke in strokes:
    #     telemetry(stroke)
    # plot.plot_dynamics(raw)


if __name__ == "__main__":
    for (dirpath, dirnames, filenames) in walk('strokes'):
        for filename in filenames:
            if not filename.endswith('npy'): continue
            if not filename == 'U+3075.npy': continue
            process(filename)

# raw = np.load('strokes/U+3092.npy')
# strokes = split_into_strokes(raw)

# for stroke in strokes:
#     metrics(stroke)
#     # print('line', np.asarray(line))


# composed = compose_from_strokes(strokes)
# # plot.plot_dynamics(composed)



# # for stroke in strokes:
# #     print('#points:', len(stroke)) # n point
# #     timestamps = stroke[:,3]
# #     timestamp_diffs = np.diff(timestamps)

# #     # Normalized sum of timestamp differences;
# #     # starting with (close to) 0.
# #     elapsed = np.cumsum(timestamp_diffs)
# #     xs = stroke[:,1]
# #     ys = stroke[:,2]
# #     x_diffs = xs[1:] - xs[:-1] # n - 1
# #     y_diffs = ys[1:] - ys[:-1] # n - 1

# #     # euclidian distances of consecutive points: n - 1
# #     dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)

# #     velocities = dists / timestamp_diffs * 1000
# #     print(velocities)