import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
# plt.rcParams['font.family'] = 'YuKyokasho'


def kinematics(title, kinematics):
    TIMESTAMP, PRESSURE, VELOCITY, ANGLE = 0, 1, 2, 3
    timestamp = kinematics[:, TIMESTAMP]
    pressure   = kinematics[:, PRESSURE]
    velocity   = kinematics[:, VELOCITY]
    angle      = kinematics[:, ANGLE]

    pressure_masked = np.ma.masked_where(pressure == 0, pressure)
    velocity_masked = np.ma.masked_where(pressure == 0, velocity)
    angle_masked = np.ma.masked_where(pressure == 0, angle)

    fig, ax1 = plt.subplots(figsize=(20, 5))
    axes = [ax1, ax1.twinx(), ax1.twinx()]

    plot_configs = [
        dict(color="tab:blue", label="Velocity", data=velocity_masked),
        dict(color="tab:red",  label="Angle",    data=angle_masked),
        dict(color="tab:gray", label="Pressure", data=pressure_masked, linestyle="-.", pad=15.0)
        # dict(color="tab:gray", label="Pressure", data=pressure, linestyle="-.", pad=15.0)
    ]

    lines = []
    for i, c in enumerate(plot_configs):
        linestyle = c.get('linestyle', None)
        pad = c.get('pad', 0.0)
        axes[i].set_ylabel(c['label'], color=c['color'], fontweight="bold")
        axes[i].tick_params(axis="y", labelcolor=c['color'], pad=pad)
        line = axes[i].plot(timestamp, c['data'], color=c['color'], label=c['label'], lw=1, linestyle=linestyle)
        lines.extend(line)

    labels = [line.get_label() for line in lines]
    plt.title(f"Kinematics: {title}", fontsize=14)
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
        idx = np.searchsorted(timestamp, mouse_x)
        idx = np.clip(idx, 0, len(timestamp) - 1)

        if idx > 0 and abs(mouse_x - timestamp[idx - 1]) < abs(mouse_x - timestamp[idx]):
            idx -= 1

        # Extract precise data from the raw arrays directly
        current_timestamp = timestamp[idx]
        current_velocity = velocity[idx]
        current_angle = angle[idx]
        current_pressure = pressure[idx]

        # Update vertical cursor line
        v_line.set_xdata([current_timestamp, current_timestamp])
        v_line.set_visible(True)

        # Structure text string safely
        tooltip_text = (
            f"Time:     {current_timestamp:.2f} ({idx})\n"
            f"Velocity: {current_velocity:.2f}\n"
            f"Angle:    {current_angle:.2f}\n"
            f"Pressure: {current_pressure:.2f}"
        )

        tooltip.set_text(tooltip_text)

        # Place the tooltip at the current point's x-coordinate,
        # but safely aligned to the vertical center of ax1's viewport
        y_min, y_max = ax1.get_ylim()
        tooltip.xy = (current_timestamp, (y_min + y_max) / 2)
        tooltip.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    fig.tight_layout()
    plt.show()
