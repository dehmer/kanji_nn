import numpy as np
from functools import cached_property, partial
import matplotlib.pyplot as plt
from ..math import calc_kinematic

class Kinematics:
    def __init__(self, raw, configs, figsize=(14, 10)):
        self.raw = raw
        self.configs = configs

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize)
        plt.subplots_adjust(left=0.08, right=0.90, bottom=0.15, wspace=0.3)

        self.fig = fig
        self.ax_top = ax_top
        self.ax_bottom = ax_bottom

        self.blocks = calc_kinematic({'raw': raw})

    @cached_property
    def timestamp(self):
        return self.blocks['raw'][:, 0]

    @cached_property
    def x(self):
        return self.blocks['raw'][:, 1]

    @cached_property
    def y(self):
        return self.blocks['raw'][:, 2]

    @cached_property
    def pressure(self):
        return self.blocks['raw'][:, 3]

    @cached_property
    def pen(self):
        return self.blocks['raw'][:, 4]

    def _plot_character(self):
        pen = self.pen
        x = self.x
        y = self.y

        self.ax_top.set_xlim(0, 1)
        self.ax_top.set_ylim(0, 1)
        self.ax_top.set_aspect('equal', adjustable='box')
        self.ax_top.invert_yaxis()  # Invert Y if image coordinates start from top-left
        self.ax_top.grid(True, linestyle=":", alpha=0.5)

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
            self.ax_top.plot(x[segment], y[segment], color="gray", alpha=0.5, zorder=1)

            # Color markers based on position inside the stroke sequence
            for idx, global_idx in enumerate(segment):
                if idx == 0:                  color, alpha = "blue", 1
                elif idx == len(segment) - 1: color, alpha = "green", 1
                else:                         color, alpha = "red", 0.2

                self.ax_top.scatter(x[global_idx], y[global_idx], color=color,
                                alpha=alpha, edgecolors="none", s=50, zorder=2)

        # Dynamic indicators:
        hover_line_h = self.ax_top.axhline(color="black", linestyle=":", alpha=0.5, visible=False)
        hover_line_v = self.ax_top.axvline(color="black", linestyle=":", alpha=0.5, visible=False)
        active_marker, = self.ax_top.plot([], [], 'ko', markersize=8, zorder=3, visible=False)

        # Dynamic Tooltip Text Box
        tooltip = self.ax_top.text(1.2, 0.036, "", transform=self.ax_top.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                            verticalalignment='bottom', fontsize=9, zorder=4)
        tooltip.set_visible(False)

        self.top_line_h = hover_line_h
        self.top_line_v = hover_line_v
        self.stroke_marker = active_marker
        self.tooltip = tooltip


    def _plot_kinematics(self):
        timestamp = self.timestamp
        pen = self.pen

        self.ax_bottom.set_title("Kinematics")
        self.ax_bottom.set_xlabel("Timestamp (ms)")

        # Mask out NaN values and pen-up movements (pen == 0)
        invalid_mask = (pen == 0)

        # Create additional axes:
        axes = [self.ax_bottom]
        for i in range(len(self.configs) - 1):
            axes.append(self.ax_bottom.twinx())

        for i, config in enumerate(self.configs):
            ax = axes[i]
            color = config['color']
            label = config['label']
            data = config['data'](self.blocks)
            masked = np.where(invalid_mask, np.nan, data)
            linestyle = config.get('linestyle', None)

            ax.set_ylabel(label, color=color)
            ax.tick_params(axis="y", labelcolor=color)
            ax.plot(timestamp, masked, color=color, label=label, lw=1, linestyle=linestyle)

        # Dynamic indicators:
        bottom_line_v = self.ax_bottom.axvline(color="black", linestyle=":", alpha=0.5, visible=False)
        create_marker = lambda ax: ax.plot([], [], 'ro', markersize=4, visible=False)[0]
        markers = [create_marker(ax) for ax in axes]

        bottom_lines_h = []
        for i, config in enumerate(self.configs):
            line_h = axes[i].axhline(color="black", linestyle=":", alpha=0.5, visible=False)
            bottom_lines_h.append(line_h)

        self.axes = axes
        self.invalid_mask = invalid_mask
        self.markers = markers
        self.bottom_line_v = bottom_line_v
        self.bottom_lines_h = bottom_lines_h


    def _update_tooltip_and_markers(self, idx):
        timestamp = self.timestamp
        x, y = self.x, self.y

        current_timestamp = timestamp[idx]
        current_x = x[idx]
        current_y = y[idx]

        # Prepare text for tooltip and update markers:
        texts = [f"Time: {current_timestamp:.0f} ms"]
        for i, config in enumerate(self.configs):
            label = config['label']
            data = config['data'](self.blocks)
            current_data = data[idx]
            text = f"{label}: {current_data:.4f}" if not np.isnan(current_data) else "NaN"
            texts.append(text)

            self.bottom_lines_h[i].set_ydata([current_data])
            self.bottom_lines_h[i].set_visible(True)

            if not self.invalid_mask[idx]:
                self.markers[i].set_data([current_timestamp], [current_data])
                self.markers[i].set_visible(True)
            else:
                self.markers[i].set_visible(False)


        self.tooltip.set_text("\n".join(texts))
        self.tooltip.set_visible(True)

        # 3. Highlight corresponding point in black on the stroke plot
        self.stroke_marker.set_data([current_x], [current_y])
        self.stroke_marker.set_visible(True)

        self.bottom_line_v.set_xdata([current_timestamp])
        self.bottom_line_v.set_visible(True)


    def _on_top_event(self, event):
        blocks, configs = self.blocks, self.configs
        timestamp = self.timestamp
        x, y = self.x, self.y

        # 1. Update horizontal and vertical tracking crosshairs
        self.top_line_h.set_ydata([event.ydata])
        self.top_line_v.set_xdata([event.xdata])
        self.top_line_h.set_visible(True)
        self.top_line_v.set_visible(True)

        # Find the nearest valid data index using Euclidean distance in space
        distances = np.sqrt((x - event.xdata)**2 + (y - event.ydata)**2)
        idx = np.argmin(distances)
        self._update_tooltip_and_markers(idx)


    def _on_bottom_event(self, event):
        timestamp = self.timestamp

        mouse_x = event.xdata
        idx = np.searchsorted(timestamp, mouse_x)
        idx = np.clip(idx, 0, len(timestamp) - 1)

        if idx > 0 and abs(mouse_x - timestamp[idx - 1]) < abs(mouse_x - timestamp[idx]):
            idx -= 1

        self._update_tooltip_and_markers(idx)


    def _event_handler(self):
        def on_event(event):
            if event.inaxes == self.ax_top:
                self._on_top_event(event)
            elif event.inaxes in self.axes:
                self._on_bottom_event(event)
            else:
                # Hide interactive indicators when mouse leaves the left plot area
                self.top_line_h.set_visible(False)
                self.top_line_v.set_visible(False)
                self.stroke_marker.set_visible(False)
                self.bottom_line_v.set_visible(False)

                for marker in self.markers:
                    marker.set_visible(False)

                for line_h in self.bottom_lines_h:
                    line_h.set_visible(False)

                self.tooltip.set_visible(False)
            self.fig.canvas.draw_idle()

        return on_event


    def show(self):
        self._plot_character()
        self._plot_kinematics()

        # Bind the hover callback logic to the Matplotlib motion event
        self.fig.canvas.mpl_connect('motion_notify_event', self._event_handler())
        plt.show()
