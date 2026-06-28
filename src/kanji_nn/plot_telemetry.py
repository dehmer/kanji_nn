import numpy as np
import matplotlib.pyplot as plt

def plot_telemetry(telemetry):
    TIME_STAMP, PRESSURE, VELOCITY, ANGLE = 0, 1, 2, 3
    time_stamp = telemetry[:, TIME_STAMP]
    pressure   = telemetry[:, PRESSURE]
    velocity   = telemetry[:, VELOCITY]
    angle      = telemetry[:, ANGLE]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    axes = [ax1, ax1.twinx(), ax1.twinx()]

    plot_configs = [
        dict(color="tab:blue", label="Velocity", data=velocity),
        dict(color="tab:red",  label="Angle",    data=angle),
        dict(color="tab:gray", label="Pressure", data=pressure, linestyle="-.", pad=15.0)
    ]

    lines = []
    for i, c in enumerate(plot_configs):
        linestyle = c.get('linestyle', None)
        pad = c.get('pad', 0.0)
        axes[i].set_ylabel(c['label'], color=c['color'], fontweight="bold")
        axes[i].tick_params(axis="y", labelcolor=c['color'], pad=pad)
        line = axes[i].plot(time_stamp, c['data'], color=c['color'], label=c['label'], lw=1, linestyle=linestyle)
        lines.extend(line)

    labels = [line.get_label() for line in lines]
    plt.title(f"Writing Telemetry", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Time (ms)", fontweight="bold")
    ax1.legend(lines, labels, loc="upper right")

    # --- Interactive Hover Elements ---
    v_line = ax1.axvline(color="gray", linestyle=":", alpha=0.8, visible=False)
    bbox_props = dict(fc="yellow", ec="black", lw=1)

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
        if event.inaxes not in axes:
            if v_line.get_visible():
                v_line.set_visible(False)
                tooltip.set_visible(False)
                fig.canvas.draw_idle()
            return

        mouse_x = event.xdata

        # Safely extract index from the main numpy timestamp array
        idx = np.searchsorted(time_stamp, mouse_x)
        idx = np.clip(idx, 0, len(time_stamp) - 1)

        if idx > 0 and abs(mouse_x - time_stamp[idx - 1]) < abs(mouse_x - time_stamp[idx]):
            idx -= 1

        # Extract precise data from the raw arrays directly
        current_time_stamp = time_stamp[idx]
        current_velocity = velocity[idx]
        current_angle = angle[idx]

        # Update vertical cursor line
        v_line.set_xdata([current_time_stamp, current_time_stamp])
        v_line.set_visible(True)

        # Structure text string safely
        tooltip_text = (
            f"Time:     {current_time_stamp:.2f}\n"
            f"Velocity: {current_velocity:.2f}\n"
            f"Angle:    {current_angle:.2f}"
        )

        tooltip.set_text(tooltip_text)

        # Place the tooltip at the current point's x-coordinate,
        # but safely aligned to the vertical center of ax1's viewport
        y_min, y_max = ax1.get_ylim()
        tooltip.xy = (current_time_stamp, (y_min + y_max) / 2)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.tight_layout()
    plt.show()
