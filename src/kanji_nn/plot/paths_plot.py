import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from svg.path import CubicBezier, Line, Move, Close
from kanji_nn.svg.bezier_obb import bezier_obb

def paths_plot(paths: list, xdim=109, ydim=109, show_obb=True, show_badges=True):
    """
    Plots pre-parsed svg.path.Path objects, their OBB patches, and
    annotates each cubic segment with its H/W straightness ratio.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Standard KanjiVG 109x109 canvas limits
    ax.set_xlim(0, xdim)
    ax.set_ylim(ydim, 0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', color='#cccccc', alpha=0.6)

    stroke_color = '#2c3e50'
    obb_color = '#4444cc'
    text_color = '#27ae60'  # Clean green for the numeric metrics

    for stroke in paths:
        mpl_verts = []
        mpl_codes = []

        for segment_idx, segment in enumerate(stroke):
            if isinstance(segment, Move):
                mpl_verts.append((segment.start.real, segment.start.imag))
                mpl_codes.append(MplPath.MOVETO)

            elif isinstance(segment, Line):
                mpl_verts.append((segment.end.real, segment.end.imag))
                mpl_codes.append(MplPath.LINETO)

            elif isinstance(segment, CubicBezier):
                # 1. Map Bezier control points
                mpl_verts.extend([
                    (segment.control1.real, segment.control1.imag),
                    (segment.control2.real, segment.control2.imag),
                    (segment.end.real, segment.end.imag)
                ])
                mpl_codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

                if show_obb:
                    # 2. Extract OBB and calculate ratio
                    obb_data = bezier_obb(segment)
                    width = obb_data['width']
                    height = obb_data['height']
                    ratio = obb_data['ratio']

                    # 3. Add the OBB bounding box patch
                    obb_patch = patches.Polygon(
                        obb_data['polygon'],
                        closed=True,
                        facecolor=obb_color,
                        edgecolor=obb_color,
                        alpha=0.50,
                        linestyle='--',
                        linewidth=0.8
                    )
                    ax.add_patch(obb_patch)

                    # Place a small badge showing the H/W ratio
                    if show_badges:
                        # 4. Label the curve with its ratio
                        # Calculate a rough midpoint of the curve using t=0.5
                        p0, p1, p2, p3 = segment.start, segment.control1, segment.control2, segment.end
                        mid_point = 0.125*p0 + 0.375*p1 + 0.375*p2 + 0.125*p3
                        ax.text(
                            mid_point.real, mid_point.imag,
                            f"{segment_idx}: {ratio:.2f}" if ratio else 'N/A',
                            color=text_color,
                            fontsize=8,
                            fontweight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=text_color, lw=0.5, alpha=0.85)
                    )

            elif isinstance(segment, Close):
                mpl_codes.append(MplPath.CLOSEPOLY)

        # Draw the compiled text path stroke
        if mpl_verts:
            mpl_path = MplPath(mpl_verts, mpl_codes)
            stroke_patch = patches.PathPatch(
                mpl_path,
                facecolor='none',
                edgecolor=stroke_color,
                linewidth=1.5,
                capstyle='round',
                joinstyle='round'
            )
            ax.add_patch(stroke_patch)

    plt.title("KanjiVG Base-Grid With H/W Curviness Indicators", fontsize=11)
    plt.tight_layout()
    plt.show()
