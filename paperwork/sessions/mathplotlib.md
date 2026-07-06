Topic: mathplotlib
Bot: Gemini
Date: 26 07 01

> Given is a Japenese Kanji character as a 2D NumPy array of strokes which contains
all points for all strokes and additional data for the points. The columns are
0: timestamp for point (in milliseconds)
1: x coordinate in the range [0, 1]
2: y coordinate in the range [0, 1]
3: delta x: difference to previous point or np.nan (in x)
4: delta y: difference to previous point or np.nan (in y)
5: distance: euklidean distance to previous point (or np.nan)
6: heading of vector from point i and i - 1 (or np.nan)
7: pen-down/pen-up status: 1 = pen-down, 0 = pen-up

Please give me an interactive mathplotlib plot with two subplots. The left plot shows
the strokes in x, y. A stroke contains round, slightly transparent markers,
red for inner points, blue for first point and green for last point.
The right plot has timestamp as x-axis and combines distance and heading as
y-axes with individual ticks and labels. Mask out y-values of np.nan and
where the pen is lifted (pen status = 0) to clearly show segments between
strokes. Interaction: Hovering the mouse cursor over the left plot should:
1. Draw a vertical and horizontal dotted line in left plot.
2. Display a dynamic tooltip with current timestamp, distance and heading values nearest to cursor.
3. For the current timestamp, mark the corresponding point in the left stroke plot as a black marker.

Here is a complete, interactive Matplotlib solution using matplotlib.widgets and event handling (motion_notify_event).

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

def plot_interactive_kanji(data):
    """
    Plots interactive Kanji stroke data.

    Parameters:
    data (np.ndarray): 2D array of shape (N, 8) with columns:
                       0: t, 1: x, 2: y, 3: dx, 4: dy, 5: dist, 6: head, 7: pen
    """
    # Extract data columns
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    dist = data[:, 5]
    head = data[:, 6]
    pen = data[:, 7]

    # -------------------------------------------------------------------------
    # Setup Figure and Axes
    # -------------------------------------------------------------------------
    fig, (ax_left, ax_right_dist) = plt.subplots(1, 2, figsize=(14, 6))
    ax_right_head = ax_right_dist.twinx()  # Shared x-axis for heading

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.15, wspace=0.3)

    # -------------------------------------------------------------------------
    # Left Plot: Kanji Strokes
    # -------------------------------------------------------------------------
    ax_left.set_title("Kanji Strokes (X, Y)")
    ax_left.set_xlabel("X coordinate")
    ax_left.set_ylabel("Y coordinate")
    ax_left.set_xlim(-0.05, 1.05)
    ax_left.set_ylim(-0.05, 1.05)
    ax_left.invert_yaxis()  # Invert Y if image coordinates start from top-left
    ax_left.grid(True, linestyle=":", alpha=0.5)

    # Identify continuous stroke segments
    stroke_indices = np.where(pen == 1)[0]
    if len(stroke_indices) == 0:
        print("No valid pen-down data to plot.")
        return

    # Find breaks where pen lifts or indices jump
    split_indices = np.where(np.diff(stroke_indices) > 1)[0] + 1
    segments = np.split(stroke_indices, split_indices)

    # Plot each segment
    for seg in segments:
        if len(seg) == 0:
            continue
        # Draw stroke path line
        ax_left.plot(x[seg], y[seg], color="gray", alpha=0.5, zorder=1)

        # Color markers based on position inside the stroke sequence
        for idx, global_idx in enumerate(seg):
            if idx == 0:
                color = "blue"   # First point
            elif idx == len(seg) - 1:
                color = "green"  # Last point
            else:
                color = "red"    # Inner points

            ax_left.scatter(x[global_idx], y[global_idx], color=color,
                            alpha=0.6, edgecolors="none", s=50, zorder=2)

    # Dynamic cursor indicators on the left plot
    hover_line_h = ax_left.axhline(color="black", linestyle=":", alpha=0.5, visible=False)
    hover_line_v = ax_left.axvline(color="black", linestyle=":", alpha=0.5, visible=False)
    active_marker, = ax_left.plot([], [], 'ko', markersize=8, zorder=3, visible=False)

    # -------------------------------------------------------------------------
    # Right Plot: Kinematics (Distance & Heading vs Time)
    # -------------------------------------------------------------------------
    ax_right_dist.set_title("Kinematics Over Time")
    ax_right_dist.set_xlabel("Timestamp (ms)")
    ax_right_dist.set_ylabel("Distance", color="tab:blue")
    ax_right_head.set_ylabel("Heading (rad)", color="tab:orange")

    ax_right_dist.tick_params(axis='y', labelcolor="tab:blue")
    ax_right_head.tick_params(axis='y', labelcolor="tab:orange")

    # Mask out NaN values and pen-up movements (pen == 0)
    invalid_mask = np.isnan(dist) | np.isnan(head) | (pen == 0)

    t_masked = np.where(invalid_mask, np.nan, t)
    dist_masked = np.where(invalid_mask, np.nan, dist)
    head_masked = np.where(invalid_mask, np.nan, head)

    # Plot masked data
    ax_right_dist.plot(t_masked, dist_masked, color="tab:blue", label="Distance", lw=1.5)
    ax_right_head.plot(t_masked, head_masked, color="tab:orange", label="Heading", lw=1.5)

    # Dynamic indicator elements for the right plot
    right_v_line1 = ax_right_dist.axvline(color="purple", linestyle="--", alpha=0.7, visible=False)
    right_point_dist, = ax_right_dist.plot([], [], 'ro', markersize=6, visible=False)
    right_point_head, = ax_right_head.plot([], [], 'ro', markersize=6, visible=False)

    # Dynamic Tooltip Text Box
    tooltip = ax_left.text(0.02, 0.02, "", transform=ax_left.transAxes,
                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'),
                           verticalalignment='bottom', fontsize=9, zorder=4)
    tooltip.set_visible(False)

    # -------------------------------------------------------------------------
    # Interaction & Hover Callback Logic
    # -------------------------------------------------------------------------
    def on_hover(event):
        # Only trigger interaction if the mouse is inside the left plot canvas
        if event.inaxes == ax_left:
            # 1. Update horizontal and vertical tracking crosshairs
            hover_line_h.set_ydata([event.ydata])
            hover_line_v.set_xdata([event.xdata])
            hover_line_h.set_visible(True)
            hover_line_v.set_visible(True)

            # Find the nearest valid data index using Euclidean distance in space
            distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            nearest_idx = np.argmin(distances)

            # Extract tracking variables
            curr_t = t[nearest_idx]
            curr_x = x[nearest_idx]
            curr_y = y[nearest_idx]
            curr_dist = dist[nearest_idx]
            curr_head = head[nearest_idx]
            curr_pen = pen[nearest_idx]

            # 2. Update Tooltip text with raw or masked metadata
            dist_str = f"{curr_dist:.4f}" if not np.isnan(curr_dist) else "NaN"
            head_str = f"{curr_head:.4f}" if not np.isnan(curr_head) else "NaN"
            pen_str = "Down" if curr_pen == 1 else "Up"

            tooltip.set_text(
                f"Time: {curr_t:.0f} ms\n"
                f"Dist: {dist_str}\n"
                f"Head: {head_str}\n"
                f"Pen: {pen_str}"
            )
            tooltip.set_visible(True)

            # 3. Highlight corresponding point in black on the stroke plot
            active_marker.set_data([curr_x], [curr_y])
            active_marker.set_visible(True)

            # Cross-plot Interaction: Highlight values in the right plot simultaneously
            right_v_line1.set_xdata([curr_t])
            right_v_line1.set_visible(True)

            if not invalid_mask[nearest_idx]:
                right_point_dist.set_data([curr_t], [curr_dist])
                right_point_head.set_data([curr_t], [curr_head])
                right_point_dist.set_visible(True)
                right_point_head.set_visible(True)
            else:
                right_point_dist.set_visible(False)
                right_point_head.set_visible(False)

            fig.canvas.draw_idle()

        else:
            # Hide interactive indicators when mouse leaves the left plot area
            hover_line_h.set_visible(False)
            hover_line_v.set_visible(False)
            active_marker.set_visible(False)
            right_v_line1.set_visible(False)
            right_point_dist.set_visible(False)
            right_point_head.set_visible(False)
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

    # Bind the hover callback logic to the Matplotlib motion event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()
```

```python

# -------------------------------------------------------------------------
# Example Usage Simulation
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # Generate 150 dummy structural data points to simulate 3 unique strokes
    np.random.seed(42)
    N = 150
    mock_data = np.zeros((N, 8))

    # Generate mock timestamps, coordinates, and continuous segment structures
    mock_data[:, 0] = np.linspace(0, 3000, N)               # column 0: timestamps
    mock_data[:, 1] = np.sin(np.linspace(0, 5, N)) * 0.4 + 0.5 # column 1: X space
    mock_data[:, 2] = np.cos(np.linspace(0, 5, N)) * 0.4 + 0.5 # column 2: Y space

    # Compute derivative transformations (dx, dy, distance, heading)
    dx = np.diff(mock_data[:, 1], prepend=np.nan)
    dy = np.diff(mock_data[:, 2], prepend=np.nan)
    mock_data[:, 3] = dx
    mock_data[:, 4] = dy
    mock_data[:, 5] = np.sqrt(dx**2 + dy**2)                # column 5: distance
    mock_data[:, 6] = np.arctan2(dy, dx)                    # column 6: heading

    # Establish distinct strokes using simulated pen lifting states
    mock_data[:, 7] = 1 # Initialize pen down
    mock_data[45:55, 7] = 0   # First lift (transition between stroke 1 and 2)
    mock_data[100:110, 7] = 0 # Second lift (transition between stroke 2 and 3)

    # Introduce explicit NaN gaps to match physical structure criteria
    mock_data[mock_data[:, 7] == 0, 3:7] = np.nan
    mock_data[0, 3:7] = np.nan

    # Run the dynamic plotting dashboard
    plot_interactive_kanji(mock_data)
```

## Key Implementation Details

* Stroke Color Assignment: The code splits the points into continuous index arrays based on pen == 1. For each isolated array, the first point (idx == 0) is forced to blue, the last point is forced to green, and all values in between remain red.
* Y-Axis Inversion: ax_left.invert_yaxis() is implemented since handwriting digitizers and canvas coordinates traditionally treat the top-left corner as origin $(0,0)$.
* Masked Line Slicing: By using np.where(invalid_mask, np.nan, ...) Matplotlib automatically isolates points mapped to np.nan values or lift occurrences, generating separate visual sequences instead of an unbroken zig-zag line.
* Interactivity Performance: The rendering utilizes fig.canvas.draw_idle() inside the listener loop to guarantee updating frame rates without dropping inputs during background calculations.

Would you like help with exporting these data stats to a CSV file, or processing real Kanji coordinates from an external database?

import numpy as np
# Test array generation logic to ensure code snippet works cleanlydata = np.zeros((10, 8))
data[:, 0] = np.arange(0, 100, 10) # timestamps
data[:, 1] = np.random.rand(10)   # x
data[:, 2] = np.random.rand(10)   # y
data[:, 7] = 1.0                  # pen down

print("Shape:", data.shape)
