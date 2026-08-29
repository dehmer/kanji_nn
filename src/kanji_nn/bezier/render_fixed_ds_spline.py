import numpy as np
from PIL import Image, ImageDraw


def render_fixed_ds_spline(spline, img_size=(500, 500), ds=0.006, steps=50):
    """
    Renders an (N, 8) array of normalized Bezier segments using a fixed delta arc-length (ds).
    """
    # 1. Setup the 50 dense parametric micro-steps per segment
    t_dense = np.linspace(0, 1, steps + 1)

    # Pre-compute Bernstein polynomials for the micro-steps
    b0 = (1 - t_dense) ** 3
    b1 = 3 * (1 - t_dense) ** 2 * t_dense
    b2 = 3 * (1 - t_dense) * t_dense ** 2
    b3 = t_dense ** 3

    # Initialize the PIL canvas
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)

    # 2. Iterate through each segment row
    for i in range(spline.shape[0]):
        # Unpack control points for the current segment
        p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y = spline[i]

        # Generate the dense micro-step coordinates
        x_dense = p0x * b0 + p1x * b1 + p2x * b2 + p3x * b3
        y_dense = p0y * b0 + p1y * b1 + p2y * b2 + p3y * b3

        # 3. Compute linear distances between micro-steps
        dx = np.diff(x_dense)
        dy = np.diff(y_dense)
        step_lengths = np.sqrt(dx**2 + dy**2)

        # Build cumulative arc-length lookup table
        s = np.zeros_like(t_dense)
        s[1:] = np.cumsum(step_lengths)

        total_length = s[-1]
        if total_length == 0:
            continue

        # 4. Generate target distances separated by exactly 'ds'
        s_uniform = np.arange(0, total_length, ds)

        # Append the absolute endpoint to prevent gaps between segments
        if len(s_uniform) == 0 or s_uniform[-1] < total_length:
            s_uniform = np.append(s_uniform, total_length)

        # Map uniform distances back to parametric t-values
        t_uniform = np.interp(s_uniform, s, t_dense)

        # 5. Evaluate final coordinates at the uniform t-values
        b0_u = (1 - t_uniform) ** 3
        b1_u = 3 * (1 - t_uniform) ** 2 * t_uniform
        b2_u = 3 * (1 - t_uniform) * t_uniform ** 2
        b3_u = t_uniform ** 3

        x_uniform = p0x * b0_u + p1x * b1_u + p2x * b2_u + p3x * b3_u
        y_uniform = p0y * b0_u + p1y * b1_u + p2y * b2_u + p3y * b3_u

        # 6. Scale normalized [0, 1] coordinates to PIL pixel space
        x_pixels = x_uniform * img_size[0]
        y_pixels = y_uniform * img_size[1]

        # Convert to a flat list of coordinate tuples for PIL
        points = list(zip(x_pixels, y_pixels))

        # Render the line segment
        draw.line(points, fill="green", width=2, joint="round")

    return img
