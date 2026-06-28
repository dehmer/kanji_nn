import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_stroke(strokes, figsize):

    # For n strokes lookup n-1 split indices.
    # Split happens at index, so add 1 to each index.
    split_indices = np.where(strokes[:, 2] == 0)[0] + 1
    lines = np.split(strokes, split_indices[:-1]) # use n-1 indices, drop last

    figure = plt.figure(figsize=figsize)
    for i, stroke in enumerate(lines):
        line = np.array(stroke)
        plt.plot(line[:, 0], line[:, 1], color='black', zorder=1)
        plt.scatter(line[:, 0], line[:, 1], marker='o', color='red', zorder=2, alpha=.3)
        plt.scatter(line[0, 0], line[0, 1], marker='o', color='green', zorder=3)
        plt.scatter(line[-1, 0], line[-1, 1], marker='o', color='blue', zorder=3)

    # Kanji coordinate systems usually start at the top-left (invert Y-axis)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xlim([0, 1]) # assume fixed x values [0, 1]
    ax.set_ylim([1, 0]) # assume fixed y values [0, 1]
    ax.set_axis_off()

    return figure

def save(filename, strokes, figsize):
    figure = plot_stroke(strokes, figsize)
    plt.savefig(filename)
    plt.close(figure)


def show(strokes, figsize):
    figure = plot_stroke(strokes, figsize)
    plt.show()

def normalize_array(arr):
    """Normalizes a numpy array to the range [0-1], safely handling NaNs."""
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def plot_dynamics(data):
    """Plots 2D path and normalized kinetics where colors match across charts

    and individual stroke durations are calculated and displayed.
    """
    # 1. Extract columns
    xs, ys, pen_down, ts = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Calculate global differences for kinematics
    dx, dy, dt = np.diff(xs), np.diff(ys), np.diff(ts)

    # 2. Compute metrics
    distances = np.sqrt(dx**2 + dy**2)
    dt_safe = np.where(dt == 0, np.nan, dt)
    velocities = distances / dt_safe

    vectors = np.column_stack((dx, dy))
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes_safe = np.where(magnitudes == 0, 1.0, magnitudes)
    unit_vectors = vectors / magnitudes_safe
    dot_products = np.clip(
        np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1), -1.0, 1.0
    )
    angle_changes = np.degrees(np.arccos(dot_products))

    # Mask out pen-up transitions so they don't corrupt metrics
    valid_transition = (pen_down[:-1] == 1) & (pen_down[1:] == 1)
    velocities[~valid_transition] = np.nan
    valid_angle = valid_transition[:-1] & valid_transition[1:]
    angle_changes[~valid_angle] = np.nan

    # Normalize metrics globally
    norm_velocities = normalize_array(velocities)
    norm_angles = normalize_array(angle_changes)

    # 3. Separate data into continuous stroke indices
    strokes_indices = []
    stroke_start = 0
    for i in range(1, len(data)):
        if pen_down[i - 1] == 0 or i == len(data) - 1:
            if stroke_start < i - 1:
                strokes_indices.append((stroke_start, i))
            stroke_start = i

    # Define a color map for the strokes (handles up to 4 strokes cleanly)
    stroke_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    # 4. Layout setup
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    ax_path = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])

    # Track durations for metadata display
    duration_text = ["Stroke Durations:"]
    total_writing_time = 0.0

    # 5. Plot data stroke by stroke to synchronize colors and track time
    for idx, (start, end) in enumerate(strokes_indices):
        color = stroke_colors[idx % len(stroke_colors)]

        # Calculate time duration for this specific stroke
        stroke_duration = ts[end - 1] - ts[start]
        total_writing_time += stroke_duration
        duration_text.append(f"  Stroke {idx+1}: {stroke_duration:.2f}s")

        # --- Plot 2D Path (Left) ---
        ax_path.plot(
            xs[start:end],
            ys[start:end],
            linewidth=4,
            marker="o",
            markersize=4,
            color=color,

            # Include label in legend:
            # label=f"Stroke {idx+1}",
        )

        # --- Plot Kinematics (Right) ---
        # Slices must map precisely to the global ts index window
        # Velocity exists for transitions (ts[1:] -> maps to index start+1 to end)
        ax_metrics.plot(
            ts[start + 1 : end],
            norm_velocities[start : end - 1],
            color=color,
            linewidth=2.5,
            label=f"S{idx+1} Velocity" if idx == 0 else "",
        )

        # Angle change requires 3 points (ts[1:-1] -> maps to index start+1 to end-1)
        if (end - start) > 2:
            ax_metrics.plot(
                ts[start + 1 : end - 1],
                norm_angles[start : end - 2],
                color=color,
                linewidth=2.0,
                linestyle="--",
                label=f"S{idx+1} Angle Change" if idx == 0 else "",
            )

    # 6. Add Vertical Markers for Pen-Up states
    in_pen_up = False
    pen_up_start = 0.0
    for i in range(len(data)):
        if pen_down[i] == 0 and not in_pen_up:
            pen_up_start = ts[i]
            in_pen_up = True
        elif (pen_down[i] == 1 or i == len(data) - 1) and in_pen_up:
            pen_up_end = ts[i]
            ax_metrics.axvspan(
                pen_up_start,
                pen_up_end,
                color="lightgray",
                alpha=0.35,
                label=(
                    "Pen Up"
                    if "Pen Up" not in ax_metrics.get_legend_handles_labels()[1]
                    else ""
                ),
            )
            in_pen_up = False

    # 7. Final Canvas Styling & Text Overlay
    ax_path.set_title("Handwriting Mechanics", fontsize=14, pad=10)
    ax_path.set_xlabel("X Coordinate")
    ax_path.set_ylabel("Y Coordinate")
    ax_path.invert_yaxis()
    ax_path.grid(True, linestyle=":")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.legend(loc="lower left")

    ax_metrics.set_title(
        "Normalized Stroke Dynamics [0-1]", fontsize=14, pad=10
    )
    ax_metrics.set_xlabel("Timestamp (seconds)")
    ax_metrics.set_ylabel("Normalized Scale")
    ax_metrics.set_ylim(-0.05, 1.05)
    ax_metrics.grid(True, linestyle=":")

    # Create custom legend entries for the right plot to keep it clean
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="gray", linewidth=2.5, linestyle="-"),
        Line2D([0], [0], color="gray", linewidth=2.0, linestyle="--"),
        Line2D([0], [0], color="lightgray", alpha=0.5, linewidth=8),
    ]

    ax_metrics.legend(
        custom_lines,
        ["Velocity (Solid)", "Angle Change (Dashed)", "Pen Up Lift"],
        loc="upper right",
    )

    # Append total time to text block and display it in the corner of the path plot
    duration_text.append(f"Total Active: {total_writing_time:.2f}s")
    text_block = "\n".join(duration_text)

    # Stroke durations:
    # ax_path.text(
    #     0.05,
    #     0.95,
    #     text_block,
    #     transform=ax_path.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    # )

    plt.tight_layout()
    plt.show()
