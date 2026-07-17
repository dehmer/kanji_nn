import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Override the font-family which supports CJK characters
matplotlib.rcParams['font.family'] = 'Hiragino Mincho ProN'


def multi_channel_plot(stroke, channels, figsize=(14, 8), tangent_length=0.30):
    """
    Creates an interactive composite visualization for a Kanji stroke.
    Displays a real-time pivoting solid tangent line tracking the mouse cursor.
    The channel subplots are seamlessly stacked vertically with zero spacing.
    A fixed HUD box prints numerical properties of the hovered frame.
    """

    num_channels = len(channels)

    # Precompute spatial tangents (dx, dy) across the entire curve trajectory
    dx = np.gradient(stroke.x)
    dy = np.gradient(stroke.y)
    magnitudes = np.hypot(dx, dy)


    # Avoid zero-division errors on stagnant points by masking
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_dx = np.where(magnitudes > 0, dx / magnitudes, 0.0)
        unit_dy = np.where(magnitudes > 0, dy / magnitudes, 0.0)

    # 1. Setup Layout Grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.3)

    # 2. Left Side: Stroke Subplot
    ax_stroke = fig.add_subplot(gs[0, 0])
    ax_stroke.plot(stroke.x, stroke.y, color='#2c3e50', linewidth=1.5, label='Stroke Path')

    # Primitives for mouse tracking: dot marker and an extended solid tangent line (tangent_length=0.30)
    stroke_marker, = ax_stroke.plot([], [], 'ro', markersize=6, alpha=0.7, zorder=5)
    tangent_line, = ax_stroke.plot([], [], color='red', linestyle='-', linewidth=2.0, alpha=0.9, zorder=4)

    title_lines = [
        "Kanji Stroke Spatial Trajectory",
        f"{stroke.literal} - {stroke.code_point}, Stroke: {stroke.stroke_index}",
        f"type: {stroke.stroke_type[1]}, {stroke.stroke_type[2]}"
    ]

    title = "\n".join(title_lines)
    ax_stroke.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax_stroke.set_xlabel('X Coordinate')
    ax_stroke.set_ylabel('Y Coordinate')
    ax_stroke.grid(True, linestyle='--', alpha=0.5)
    ax_stroke.set_xlim(0, 1)
    ax_stroke.set_ylim(0, 1)
    ax_stroke.invert_yaxis()

    ax_stroke.set_aspect('equal')

    # 3. Right Side: Vertically Stacked Channels
    gs_channels = gs[0, 1].subgridspec(num_channels, 1, hspace=0.0)
    ax_channels = []
    vlines = []

    colors = plt.cm.tab10.colors

    t = stroke.t
    t -= t[0]

    cuts = stroke.props.get('cuts', None)

    for i in range(num_channels):
        sharex = ax_channels[0] if i > 0 else None
        ax = fig.add_subplot(gs_channels[i, 0], sharex=sharex)
        ax_channels.append(ax)

        channel_data = stroke.features[channels[i]]
        ax.plot(t, channel_data, color=colors[i % len(colors)], linewidth=1)
        ax.set_ylabel(channels[i], fontsize=10, fontweight='bold')
        ax.grid(False)
        ax.grid(True, linestyle='--', alpha=0.5)

        if cuts and cuts[0] < stroke.n_points:
            ax.axvline(t[cuts[0]], color='black', linestyle='--', linewidth=1, alpha=0.6)

        if cuts and cuts[1] < stroke.n_points:
            ax.axvline(t[cuts[1] - 1], color='black', linestyle='--', linewidth=1, alpha=0.6)

        # # median
        # ax.axvline(t[t.size // 2], color='black', linewidth=1.5, alpha=1)
        vline = ax.axvline(x=np.nan, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
        vlines.append(vline)

        if i < num_channels - 1:
            ax.tick_params(bottom=False, labelbottom=False)
        else:
            ax.tick_params(bottom=True, labelbottom=True)
            ax.set_xlabel('Elapsed time (ms)', fontsize=11)

    ax_channels[0].set_title('Derived Kinematic Channels', fontsize=12, fontweight='bold', pad=10)
    fig.align_ylabels(ax_channels)

    # 4. Fixed HUD Display Box (Placed via overall figure coordinates)
    tooltip_box = fig.text(
        0.02, 0.25, "",
        fontsize=10,
        fontfamily='monospace',
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f6fa', edgecolor='#dcdde1', alpha=0.9, zorder=10)
    )
    tooltip_box.set_visible(False)

    # ---------------------------------------------------------------------------
    # 5. Bidirectional Mouse Hover Event Handling Logic
    # ---------------------------------------------------------------------------
    def on_move(event):
        if event.inaxes is None:
            return

        target_idx = None

        # Determine target point based on active axis
        if event.inaxes == ax_stroke:
            distances = np.sum((stroke.xy - np.array([[event.xdata, event.ydata]])) ** 2, axis=1)
            target_idx = np.argmin(distances)
        elif event.inaxes in ax_channels:
            target_idx = np.argmin(np.abs(t - event.xdata))

        if target_idx is not None:
            # Extract point location and its corresponding precomputed directional vector
            px, py = stroke.xy[target_idx, 0], stroke.xy[target_idx, 1]
            tx, ty = unit_dx[target_idx], unit_dy[target_idx]

            # Calculate the front and back extensions of the longer tangent line
            half_len = tangent_length / 2.0
            t_x_coords = [px - tx * half_len, px + tx * half_len]
            t_y_coords = [py - ty * half_len, py + ty * half_len]

            # Update the visuals on the Stroke canvas
            stroke_marker.set_data([px], [py])
            # tangent_line.set_data(t_x_coords, t_y_coords)

            # Update the vertical timelines on the Channel canvas
            current_time = stroke.t[target_idx]
            for vline in vlines:
                vline.set_xdata([current_time])

            # Update HUD text with right-aligned formatting
            info_text = f"--- METRICS ---\n"
            info_text += f"Time:  {current_time:.3f}\n"
            info_text += f"Index: {target_idx}\n"
            info_text += f"Pos X: {px:.3f}\n"
            info_text += f"Pos Y: {py:.3f}\n"
            info_text += f"---------------\n"

            for idx in range(num_channels):
                channel_data = stroke.features[channels[idx]]
                val = channel_data[target_idx]
                val = f"{val}" if isinstance(channel_data[target_idx], np.ndarray) else f"{val:>7.3f}"
                text = f"{channels[idx][:10]:<10}: {val}\n"
                info_text += text

            tooltip_box.set_text(info_text.strip())
            tooltip_box.set_visible(True)

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)

    return fig
