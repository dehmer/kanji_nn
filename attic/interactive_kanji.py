import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'

def _plot_character(ax, blocks):
    pen = blocks['raw'][:, 4]
    x = blocks['raw'][:, 1]
    y = blocks['raw'][:, 2]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Invert Y if image coordinates start from top-left
    ax.grid(True, linestyle=":", alpha=0.5)

    # Identify continuous stroke segments
    stroke_indices = np.where(pen == 1)[0]
    if len(stroke_indices) == 0:
        print("No valid pen-down data to plot.")
        return

    # Find breaks where pen lifts or indices jump
    split_indices = np.where(np.diff(stroke_indices) > 1)[0] + 1
    segments = np.split(stroke_indices, split_indices)

    # Plot each segment.
    for segment in segments:
        if len(segment) == 0:
            continue

        # Draw stroke path line
        ax.plot(x[segment], y[segment], color="gray", alpha=0.5, zorder=1)

        # Color markers based on position inside the stroke sequence
        for idx, global_idx in enumerate(segment):
            if idx == 0:                  color, alpha = "blue", 1
            elif idx == len(segment) - 1: color, alpha = "green", 1
            else:                         color, alpha = "red", 0.2

            ax.scatter(x[global_idx], y[global_idx], color=color,
                            alpha=alpha, edgecolors="none", s=50, zorder=2)

    # Dynamic indicators:
    hover_line_h = ax.axhline(color="black", linestyle=":", alpha=0.5, visible=False)
    hover_line_v = ax.axvline(color="black", linestyle=":", alpha=0.5, visible=False)
    active_marker, = ax.plot([], [], 'ko', markersize=8, zorder=3, visible=False)

    # Dynamic Tooltip Text Box
    tooltip = ax.text(1.2, 0.036, "", transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                        verticalalignment='bottom', fontsize=9, zorder=4)
    tooltip.set_visible(False)

    return {
        'hover_line_h': hover_line_h,
        'hover_line_v': hover_line_v,
        'active_marker': active_marker,
        'tooltip': tooltip
    }


def _plot_kinematics(ax_distance, blocks, plot_configs):
    timestamp = blocks['raw'][:, 0]
    pen = blocks['raw'][:, 4]

    ax_distance.set_title("Kinematics")
    ax_distance.set_xlabel("Timestamp (ms)")

    # Mask out NaN values and pen-up movements (pen == 0)
    invalid_mask = (pen == 0)

    # Create additional axes:
    axes = [ax_distance]
    for i in range(len(plot_configs) - 1):
        axes.append(ax_distance.twinx())

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        color = config['color']
        label = config['label']
        data = config['data'](blocks)
        masked = np.where(invalid_mask, np.nan, data)
        linestyle = config.get('linestyle', None)

        ax.set_ylabel(label, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.plot(timestamp, masked, color=color, label=label, lw=1, linestyle=linestyle)

    # Dynamic indicators:
    right_v_line1 = ax_distance.axvline(color="black", linestyle=":", alpha=0.5, visible=False)

    create_marker = lambda ax: ax.plot([], [], 'ro', markersize=4, visible=False)[0]
    markers = [create_marker(ax) for ax in axes]

    return {
        'axes': axes,
        'invalid_mask': invalid_mask,
        'markers': markers,
        'right_v_line1': right_v_line1,
    }

def event_handler(gd, cd, kd):
    blocks = gd['blocks']
    plot_configs = gd['plot_configs']

    # Extract data columns
    timestamp = blocks['raw'][:, 0]
    x = blocks['raw'][:, 1]
    y = blocks['raw'][:, 2]

    # -------------------------------------------------------------------------
    # Interaction & Hover Callback Logic
    # -------------------------------------------------------------------------
    def on_hover(event):
        # Only trigger interaction if the mouse is inside the top plot canvas
        if event.inaxes == gd['ax_top']:
            # 1. Update horizontal and vertical tracking crosshairs
            cd['hover_line_h'].set_ydata([event.ydata])
            cd['hover_line_v'].set_xdata([event.xdata])
            cd['hover_line_h'].set_visible(True)
            cd['hover_line_v'].set_visible(True)

            # Find the nearest valid data index using Euclidean distance in space
            distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
            nearest_idx = np.argmin(distances)

            # Extract tracking variables
            current_timestamp = timestamp[nearest_idx]
            current_x = x[nearest_idx]
            current_y = y[nearest_idx]

            # Prepare text for tooltip and update markers:
            texts = [f"Time: {current_timestamp:.0f} ms"]
            for i, config in enumerate(plot_configs):
                label = config['label']
                data = config['data'](blocks)
                current = data[nearest_idx]
                text = f"{label}: {current:.4f}" if not np.isnan(current) else "NaN"
                texts.append(text)

                if not kd['invalid_mask'][nearest_idx]:
                    kd['markers'][i].set_data([current_timestamp], [current])
                    kd['markers'][i].set_visible(True)
                else:
                    kd['markers'][i].set_visible(False)


            cd['tooltip'].set_text("\n".join(texts))
            cd['tooltip'].set_visible(True)

            # 3. Highlight corresponding point in black on the stroke plot
            cd['active_marker'].set_data([current_x], [current_y])
            cd['active_marker'].set_visible(True)

            # Cross-plot Interaction: Highlight values in the right plot simultaneously
            kd['right_v_line1'].set_xdata([current_timestamp])
            kd['right_v_line1'].set_visible(True)

            gd['fig'].canvas.draw_idle()

        else:
            # Hide interactive indicators when mouse leaves the left plot area
            cd['hover_line_h'].set_visible(False)
            cd['hover_line_v'].set_visible(False)
            cd['active_marker'].set_visible(False)
            kd['right_v_line1'].set_visible(False)

            for marker in kd['markers']:
                marker.set_visible(False)

            cd['tooltip'].set_visible(False)
            gd['fig'].canvas.draw_idle()

    return on_hover


def interactive_kanji(blocks, plot_configs, figsize=(14, 10)):
    """
    Plots interactive Kanji stroke data.

    Parameters:
    data (np.ndarray): 2D array of shape (N, 8) with columns:
                       0: t, 1: x, 2: y, 3: dx, 4: dy, 5: dist, 6: head, 7: pen
    """

    # -------------------------------------------------------------------------
    # Setup Figure and Axes
    # -------------------------------------------------------------------------
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.15, wspace=0.3)

    gd = {'blocks': blocks, 'plot_configs': plot_configs, 'fig': fig, 'ax_top': ax_top}
    cd = _plot_character(ax_top, blocks)
    kd = _plot_kinematics(ax_bottom, blocks, plot_configs)
    on_hover = event_handler(gd, cd, kd)

    # Bind the hover callback logic to the Matplotlib motion event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    plt.show()
