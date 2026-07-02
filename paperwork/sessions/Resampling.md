Topic: Resampling
Bot: Gimini
Date: 26 06 30

> In the context of LSTM training, what are possible resample strategies for 2D stroke data?

Resampling 2D stroke data for LSTM training requires balancing temporal dynamics with geometric shapes. Because Long Short-Term Memory (LSTM) networks require consistent sequence structures, raw digital ink data (sequences of coordinates recorded over time) must be standardized.
The main strategies fall into two categories: spatial resampling (focusing on the shape) and temporal resampling (focusing on the speed).

------------------------------
## Spatial Resampling (Equidistant Path)
This strategy removes the time variable and samples points at equal spatial distances along the stroke path. It is ideal for tasks like handwriting recognition where shape matters more than drawing speed.

* Linear Interpolation: Calculate the total length of the stroke. Divide it into N equal segments. Interpolate new points at these fixed spatial intervals.
* Douglas-Peucker Algorithm: Downsamples data by removing points that fall within a specified distance threshold (ε) from a straight line segment. This preserves sharp corners and critical geometric vertices while drastically reducing sequence length.
* Spline Interpolation: Uses cubic or B-splines to smooth the trajectory first, then samples points equidistantly along the curve. This prevents jagged artifacts caused by shaky hands or hardware sensor noise.

------------------------------
## Temporal Resampling (Fixed Time Step)
This strategy preserves the dynamics, speed, and acceleration of the drawing process by sampling points at fixed time intervals. It is ideal for biometric verification, signature authentication, or stroke-based emotion analysis.

* Fixed Frequency Decimation: Select every k-th point from the raw sensor stream (e.g., keeping every 3rd point). This assumes the original hardware sampling rate was strictly constant.
* Linear Time Interpolation: Re-index the time array to a strict uniform grid (e.g., a point every 10 milliseconds). Interpolate the X and Y coordinates at those exact timestamps.
* Zero-Padding or Truncation: Keep the raw points exactly as they were recorded. Cut off strokes that are too long, or pad shorter strokes with zeros (or mask values) up to a maximum time step T. LSTMs natively support masking layers to ignore these padded steps. [1, 2]

------------------------------
## Hybrid and Data Augmentation Strategies

* Dynamic Time Warping (DTW) Alignment: Resamples a stroke to match the length or velocity profile of a reference standard template.
* Velocity-Based Resampling: Samples more points where the pen moves slowly (curves, fine details) and fewer points where the pen moves fast (straight lines). This mimics how humans naturally allocate motor control.

------------------------------
To help narrow down the best approach for your architecture, tell me:

* What is the ultimate goal of your LSTM model? (e.g., handwriting recognition, signature verification, or generative art?)
* Are you currently dealing with variable-length sequence errors during batch training?
* Do you need to preserve the speed and pressure of the drawing, or just the final visual shape?


[1] [https://pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC12292929/)
[2] [https://pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC8901655/)

> What is Spline Interpolation?

Spline interpolation is a mathematical method that connects data points using smooth, curved polynomial functions instead of straight lines. [1, 2, 3, 4]
Instead of fitting one complex formula to all data points at once, spline interpolation fits simple, lower-degree polynomials (usually cubics) between each neighboring pair of points. These local curves are forced to meet smoothly at the data points, which are called knots. [5, 6, 7, 8, 9]

------------------------------
## Why Use Spline Interpolation?

* No Runge's Phenomenon: High-degree polynomials tend to oscillate wildly at the edges of a dataset. Splines use low-degree polynomials locally, keeping the curve stable. [10, 11, 12, 13]
* Guaranteed Smoothness: It ensures that the transitions between points are visually smooth, with no sharp corners or jagged edges. [14, 15, 16]
* Accurate Continuity: It forces the first and second mathematical derivatives (slope and acceleration) to match perfectly at every single knot. [17, 18]

------------------------------
## Common Types of Splines

* Linear Spline: Connects points with straight lines. It is fast but creates sharp corners (no smooth transitions). [19, 20, 21, 22, 23]
* Cubic Spline (Most Common): Connects each pair of points with a unique third-degree polynomial (ax³ + bx² + cx + d). It delivers a perfectly smooth curve that is highly natural to the human eye. [24, 25, 26, 27, 28]
* B-Spline (Basis Spline): The curve does not necessarily pass through all the control points; instead, the points act as anchors that pull and shape the curve toward them. [29, 30, 31, 32, 33]

------------------------------
## How It Applies to Your 2D Stroke Data
If your original digital ink data has jagged edges from a shaky hand or an imprecise stylus sensor, [Cubic Spline Interpolation](https://www.geeksforgeeks.org/machine-learning/cubic-spline-interpolation/) recalculates the path.

   1. It builds a perfectly smooth mathematical curve through your raw coordinate points.
   2. You can then use that smooth curve to sample brand-new, evenly spaced X and Y coordinates for your LSTM model. [34]

------------------------------
If you want to implement this, I can provide a quick Python code snippet using scipy.interpolate to smooth your stroke data. Would you like to see how to code it, or would you prefer a breakdown of how it compares to Linear Interpolation?

[1] [https://deepai.org](https://deepai.org/machine-learning-glossary-and-terms/interpolation)
[2] [https://www.sciencedirect.com](https://www.sciencedirect.com/topics/computer-science/interpolation-problem)
[3] [https://www.analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/06/power-of-interpolation-in-python-to-fill-missing-values/)
[4] [https://www.meegle.com](https://www.meegle.com/en_us/topics/quantization/quantization-vs-interpolation)
[5] [https://link.springer.com](https://link.springer.com/chapter/10.1007/978-3-031-40478-8_8)
[6] [https://www.mdpi.com](https://www.mdpi.com/2504-2289/9/12/308)
[7] [https://www2.rivier.edu](https://www2.rivier.edu/journal/ROAJ-Fall-2013/J784-Wang_cubic-splines.pdf)
[8] [https://www.studysmarter.co.uk](https://www.studysmarter.co.uk/explanations/engineering/engineering-mathematics/interpolation/)
[9] [https://users.tricity.wsu.edu](https://users.tricity.wsu.edu/~hudson/Teaching/EE221/Text/15%20Interpolation%20II.pdf)
[10] [https://medium.com](https://medium.com/@ronantech/exploring-interpolation-methods-for-time-series-data-lagrange-splines-rbf-and-more-fc80e7d096a2)
[11] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Polynomial_interpolation)
[12] [https://search.proquest.com](https://search.proquest.com/openview/11539ba4810b5474868b2a28c29957d8/1?pq-origsite=gscholar&cbl=2032366)
[13] [https://metricgate.com](https://metricgate.com/blogs/spline-regression-vs-linear-models/)
[14] [https://intellipaat.com](https://intellipaat.com/blog/what-is-interpolation/)
[15] [https://www.aitronik.com](https://www.aitronik.com/optimizing-autonomous-navigation-why-we-use-splines-for-real-world-trajectories/)
[16] [https://sunandoroy.org](https://sunandoroy.org/2025/05/30/understanding-interest-rate-shocks-and-spline-interpolation-in-yield-curve-analysis/)
[17] [https://www.r-bloggers.com](https://www.r-bloggers.com/2017/05/connecting-the-dots-a-quick-primer-on-cubic-splines/)
[18] [https://fiveable.me](https://fiveable.me/lists/curve-fitting-methods)
[19] [https://www.itl.nist.gov](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch3/lin_inte.pdf)
[20] [https://www.blogs.unicamp.br](https://www.blogs.unicamp.br/zero/3591/)
[21] [https://airbyte.com](https://airbyte.com/data-engineering-resources/data-interpolation)
[22] [https://ikcest-drr.data.ac.cn](https://ikcest-drr.data.ac.cn/tutorial/k2016)
[23] [https://www.linkedin.com](https://www.linkedin.com/pulse/interest-rate-shocks-spline-interpolation-dr-sunando-roy-0wcce)
[24] [https://nm.mathforcollege.com](https://nm.mathforcollege.com/NumericalMethodsTextbookUnabridged/chapter-05.05-spline-method-of-interpolation.html)
[25] [https://www.ni.com](https://www.ni.com/docs/en-US/bundle/labview/page/spline-interpolation.html)
[26] [https://www.linkedin.com](https://www.linkedin.com/posts/ibluesun_b-spline-and-catmull-rom-spline-are-two-fundamental-activity-7366103694292992000-DVUm)
[27] [https://interpolationcalculator.com](https://interpolationcalculator.com/types-of-interpolation/)
[28] [https://pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC6402144/)
[29] [https://dspace.library.uvic.ca](https://dspace.library.uvic.ca/bitstreams/b24e9142-b4a0-43a5-be87-6c02f32bf399/download)
[30] [https://en.wikipedia.org](https://en.wikipedia.org/wiki/Catmull%E2%80%93Rom_spline)
[31] [https://www.embird.net](https://www.embird.net/machine-embroidery-tutorials/studio-next/splines.htm)
[32] [https://www.sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0167839623000225)
[33] [https://support.apple.com](https://support.apple.com/en-in/guide/final-cut-pro/ver3f97041a2/mac)
[34] [https://onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/10.1155/2015/563629)

> Ok. Show me the code. The stroke data for a kanji character is given as a 2D NumPy array of variable length and 3 columns: x, y and pen-down/-up status.

Here is the Python code to resample a variable-length Kanji stroke using Cubic Spline Interpolation via the [scipy.interpolate library](https://docs.scipy.org/doc/scipy/reference/interpolate.html).
Because a Kanji character consists of multiple independent strokes, you cannot run spline interpolation across the entire array at once. Doing so would cause the math to accidentally draw smooth, sweeping curves connecting the end of one stroke to the start of the next. The code below separates the data by the pen status, smooths each stroke individually, and reconstructs the character. [1]

## Complete Python Implementation

```python
import numpy as np
from scipy.interpolate import CubicSpline

def resample_stroke_data(stroke_data, num_points_per_stroke=32):
    """
    Resamples variable-length Kanji stroke data into a fixed-size sequence
    using Cubic Spline Interpolation.

    Parameters:
    -----------
    stroke_data : numpy.ndarray
        Array of shape (N, 3) containing columns: [x, y, pen_status]
        pen_status is 0 for pen-down (drawing) and 1 for pen-up (end of stroke).
    num_points_per_stroke : int
        The fixed number of equidistant points to sample along each individual stroke.

    Returns:
    --------
    numpy.ndarray
        Resampled array of shape (M * num_points_per_stroke, 3) where M is the number of strokes.
    """
    # 1. Identify where individual strokes end based on pen-up status (1)
    # We find the indices of the pen-up events and split the array
    pen_up_indices = np.where(stroke_data[:, 2] == 1)[0]

    # Handle edge case where no pen-up is marked, treat whole thing as one stroke
    if len(pen_up_indices) == 0:
        stroke_segments = [stroke_data]
    else:
        # Split into sub-arrays after each pen-up index
        stroke_segments = np.split(stroke_data, pen_up_indices + 1)
        # Remove empty trailing splits if the data ended exactly on a pen-up
        stroke_segments = [s for s in stroke_segments if len(s) > 0]

    resampled_character = []

    for stroke in stroke_segments:
        x = stroke[:, 0]
        y = stroke[:, 1]
        n_raw_points = len(stroke)

        # LSTMs need fixed steps, but short strokes (like dots) need safety overrides
        if n_raw_points < 2:
            # Cannot interpolate 1 point; pad manually to keep shape consistency
            padded_stroke = np.repeat(stroke, num_points_per_stroke, axis=0)
            resampled_character.append(padded_stroke)
            continue

        # 2. Calculate the cumulative spatial distance along the stroke path
        # This acts as the independent variable (parameterizing the curve)
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

        # Safety check for completely identical/overlapping points to avoid division by zero
        if cumulative_distance[-1] == 0:
            cumulative_distance = np.linspace(0, 1, n_raw_points)

        # 3. Fit Cubic Splines for both X and Y dimensions independently
        # bc_type='natural' keeps the boundaries stable
        cs_x = CubicSpline(cumulative_distance, x, bc_type='natural')
        cs_y = CubicSpline(cumulative_distance, y, bc_type='natural')

        # 4. Generate perfectly equidistant spatial steps along the curve
        new_distances = np.linspace(0, cumulative_distance[-1], num_points_per_stroke)
        resampled_x = cs_x(new_distances)
        resampled_y = cs_y(new_distances)

        # 5. Reconstruct the 3-column format [x, y, pen_status]
        # Every point in this stroke gets a 0 (pen-down), except the very last one which gets a 1
        resampled_status = np.zeros(num_points_per_stroke)
        resampled_status[-1] = 1

        resampled_stroke = np.column_stack((resampled_x, resampled_y, resampled_status))
        resampled_character.append(resampled_stroke)

    # Combine all individual processed strokes back into a single matrix
    return np.vstack(resampled_character)
```

```python
# --- Example Usage Verification ---
if __name__ == "__main__":
    # Simulating a mock Kanji character with 2 strokes of different lengths
    # Stroke 1: 5 raw points, ends with pen-up (1)
    stroke_1 = np.array([
        [10.0, 10.0, 0],
        [12.0, 15.0, 0],
        [15.0, 18.0, 0],
        [18.0, 19.0, 0],
        [22.0, 20.0, 1]
    ])
    # Stroke 2: 4 raw points, ends with pen-up (1)
    stroke_2 = np.array([
        [30.0, 40.0, 0],
        [32.0, 35.0, 0],
        [34.0, 25.0, 0],
        [35.0, 10.0, 1]
    ])
    mock_kanji = np.vstack((stroke_1, stroke_2))

    # Resample so that every single stroke contains exactly 16 points
    fixed_length_data = resample_stroke_data(mock_kanji, num_points_per_stroke=16)

    print(f"Original shape: {mock_kanji.shape}")
    print(f"Resampled shape: {fixed_length_data.shape}")  # Expecting (2 * 16, 3) = (32, 3)
    print(f"Pen-up markers are at indices: {np.where(fixed_length_data[:, 2] == 1)[0]}")
```

## Key Considerations for your LSTM:

* Sequence Length Stability: Setting num_points_per_stroke=32 guarantees that a Kanji with 4 strokes always has exactly $4 \times 32 = 128$ rows. This completely eliminates variable-length errors during mini-batching if you mask or group characters by stroke count.
* Loss of Writing Velocity: Because this forces spatial equidistance, the raw velocity data ($dx/dt$) is regularized away. If your LSTM relies on how fast a person wrote a line to determine identity or emotion, you should swap out cumulative_distance for a strict time tracking array ($t$).

------------------------------
If you plan to feed this output directly into your neural network, would you like help writing a PyTorch Dataset class that uses this function, or do you need a quick way to visualize and plot the resampled strokes using Matplotlib?

[1] [https://www.chokanji.com](https://www.chokanji.com/en/ckk/manual/primer/primer-2.html)

> Please change the pen-down/pen-up semantics to value 1.0 for pen-down and 0.0 for pen-up.

Here is the updated code with the pen semantics inverted: 1.0 represents pen-down (drawing) and 0.0 represents pen-up (lifting the pen at the end of a stroke).
The splitting logic now looks for 0.0 to find the boundaries between individual strokes.
## Updated Python Implementation

```Python
import numpy as np
from scipy.interpolate import CubicSpline

def resample_stroke_data(stroke_data, num_points_per_stroke=32):
    """
    Resamples variable-length Kanji stroke data into a fixed-size sequence
    using Cubic Spline Interpolation.

    Parameters:
    -----------
    stroke_data : numpy.ndarray
        Array of shape (N, 3) containing columns: [x, y, pen_status]
        pen_status is 1.0 for pen-down (drawing) and 0.0 for pen-up (end of stroke).
    num_points_per_stroke : int
        The fixed number of equidistant points to sample along each individual stroke.

    Returns:
    --------
    numpy.ndarray
        Resampled array of shape (M * num_points_per_stroke, 3) where M is the number of strokes.
    """
    # 1. Identify where individual strokes end based on pen-up status (0.0)
    # We find the indices of the pen-up events and split the array
    pen_up_indices = np.where(stroke_data[:, 2] == 0.0)[0]

    # Handle edge case where no pen-up is marked, treat whole thing as one stroke
    if len(pen_up_indices) == 0:
        stroke_segments = [stroke_data]
    else:
        # Split into sub-arrays right after each pen-up index
        stroke_segments = np.split(stroke_data, pen_up_indices + 1)
        # Remove empty trailing splits if the data ended exactly on a pen-up
        stroke_segments = [s for s in stroke_segments if len(s) > 0]

    resampled_character = []

    for stroke in stroke_segments:
        x = stroke[:, 0]
        y = stroke[:, 1]
        n_raw_points = len(stroke)

        # Short stroke safety override (e.g., single-dot strokes)
        if n_raw_points < 2:
            padded_stroke = np.repeat(stroke, num_points_per_stroke, axis=0)
            resampled_character.append(padded_stroke)
            continue

        # 2. Calculate the cumulative spatial distance along the stroke path
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_distance = np.concatenate(([0.0], np.cumsum(distances)))

        # Safety check for overlapping points to avoid division by zero
        if cumulative_distance[-1] == 0:
            cumulative_distance = np.linspace(0, 1, n_raw_points)

        # 3. Fit Cubic Splines for both X and Y dimensions independently
        cs_x = CubicSpline(cumulative_distance, x, bc_type='natural')
        cs_y = CubicSpline(cumulative_distance, y, bc_type='natural')

        # 4. Generate perfectly equidistant spatial steps along the curve
        new_distances = np.linspace(0, cumulative_distance[-1], num_points_per_stroke)
        resampled_x = cs_x(new_distances)
        resampled_y = cs_y(new_distances)

        # 5. Reconstruct the 3-column format [x, y, pen_status]
        # Every point in this stroke gets a 1.0 (pen-down), except the final point which gets 0.0
        resampled_status = np.ones(num_points_per_stroke)
        resampled_status[-1] = 0.0

        resampled_stroke = np.column_stack((resampled_x, resampled_y, resampled_status))
        resampled_character.append(resampled_stroke)

    # Combine all processed strokes back into a single matrix
    return np.vstack(resampled_character)
```

```python
# --- Example Usage Verification ---if __name__ == "__main__":
    # Simulating a mock Kanji character with 2 strokes of different lengths
    # Stroke 1: 5 raw points, ends with pen-up (0.0)
    stroke_1 = np.array([
        [10.0, 10.0, 1.0],
        [12.0, 15.0, 1.0],
        [15.0, 18.0, 1.0],
        [18.0, 19.0, 1.0],
        [22.0, 20.0, 0.0]
    ])
    # Stroke 2: 4 raw points, ends with pen-up (0.0)
    stroke_2 = np.array([
        [30.0, 40.0, 1.0],
        [32.0, 35.0, 1.0],
        [34.0, 25.0, 1.0],
        [35.0, 10.0, 0.0]
    ])
    mock_kanji = np.vstack((stroke_1, stroke_2))

    # Resample so that every single stroke contains exactly 16 points
    fixed_length_data = resample_stroke_data(mock_kanji, num_points_per_stroke=16)

    print(f"Original shape: {mock_kanji.shape}")
    print(f"Resampled shape: {fixed_length_data.shape}")  # (2 * 16, 3) = (32, 3)
    print(f"Pen-up markers (value 0.0) are at indices: {np.where(fixed_length_data[:, 2] == 0.0)[0]}")
```

------------------------------
Next, do you want to see how to pack these variable-stroke characters into fixed-size mini-batches for training, or do you need a function to normalize the X/Y spatial scales across different characters?

> What steps are advisable on raw digital stroke input before spline interpolation?

To ensure spline interpolation yields clean, stable, and accurate results, raw digital stroke input requires preprocessing. Because cubic splines are highly sensitive to sensor anomalies, noise, and human hand tremors, jumping straight to interpolation can cause mathematical instability or introduce unwanted artifacts.
Applying the following steps in sequence will optimize your raw data before fitting the splines:
## 1. Remove Exact Duplicate Points (De-duplication)
Stylus sensors often sample at a fixed hardware frequency (e.g., 200 Hz). If the user pauses their hand or writes very slowly, the device records multiple identical or nearly identical coordinate points at the exact same spatial location.

* Why: If two adjacent points have a spatial distance of zero, calculating the cumulative distance yields an interval of zero (Δ x = 0, Δ y = 0). Passing this into a spline algorithm causes a division-by-zero error or a singular matrix crash.
* Action: Filter out consecutive points where both Δ x and Δ y fall below an extremely small threshold (e.g., 10⁻⁵).

## 2. Stroke-Level Segmentation
Raw input usually arrives as a continuous stream of points over time.

* Why: Spline interpolation relies on a continuous path. If you attempt to run a spline over the entire character sequence at once, the algorithm will mathematically connect the end of one stroke to the beginning of the next, creating artificial "ghost curves" in the air.
* Action: Split your raw data array into isolated, independent sub-arrays using your pen_status == 0.0 markers before performing any calculations.

## 3. Noise Filtering and Smoothing
Raw data from touchscreens or digital tablets often contains high-frequency noise, sensor jitter, or jagged artifacts from natural hand tremors. Splines strictly pass through every point, meaning they will perfectly replicate—and even amplify—this noise. [1]

* Why: Noise creates sharp spikes in the trajectory derivatives, leading to wild oscillations (ringing artifacts) in the resulting spline.
* Action: Apply a light Gaussian filter or a 1D Moving Average Smooth independently across the X and Y arrays of the raw stroke. This rounds out sensor jitter while preserving the macro-geometry of the Kanji.

## 4. Hook and Artifact Removal
When a writer lifts or drops a stylus, the physical contact often creates a microscopic "hook" or "tail" at the absolute beginning or end of a stroke due to the pen dragging slightly.

* Why: These tiny, accidental hooks distort the boundary conditions of the cubic spline, causing the edges of your smoothed strokes to flare out unnerving directions.
* Action: Remove the first and last points of a stroke if they exhibit a sudden, radical 90-to-180-degree directional snap over a negligible distance.

## 5. Handle Critical Short-Stroke Exceptions
A standard cubic spline requires a minimum number of unique points to define its boundary equations (typically at least 2 points for a linear boundary, or 3 to 4 points for a full natural cubic curve). [2, 3]

* Why: Kanji characters frequently contain tiny dots (ten) that may only register as 1 or 2 raw points total in the data stream. Passing these directly to a spline function will crash your code.
* Action: Implement an explicit conditional check. If a stroke segment contains fewer than 3 points, bypass the spline algorithm entirely and use simple linear padding or duplication to match your target LSTM sequence length.

------------------------------
Would you like to see how to integrate Steps 1 and 5 (De-duplication and Short-Stroke Safety) directly into your existing Python code, or do you need to look at spatial normalization techniques (like min-max scaling) to make the characters uniform before they hit the LSTM?

[1] [https://support.ptc.com](https://support.ptc.com/help/creo/ced_modeling/r20.8.0.0/en/ced_modeling/OSDM_Annotation/am_draw_geop5.html)
[2] [https://eldoradosoft.com](https://eldoradosoft.com/v10features.pdf)
[3] [https://help.fieldsystems.trimble.com](https://help.fieldsystems.trimble.com/tbc/3263.htm)

> Given is a Python list of kanji strokes as 2D NumPy arrays with x and y coordinates normalised to [0, 1] respectively. Give me a resampling function based on cubic splines and a fixed number of points per stroke, e.g. 32. Alternatively the number of resulting points may depend on the cumulated stroke length. Make sure the output coordinates are also normalised to [0, 1]. CubicSpline from scipy.interpolate may be used.

A good approach is:

1. Parameterize each stroke by its cumulative arc length.
2. Fit separate cubic splines for (x(s)) and (y(s)).
3. Sample uniformly in arc length.
4. Clip the result back to `[0, 1]` because cubic splines may slightly overshoot near the endpoints.

Using arc length instead of the original point index gives nearly uniform spacing regardless of the original sampling density.

```python
import numpy as np
from scipy.interpolate import CubicSpline


def resample_stroke(stroke, n_points=32):
    """
    Resample a stroke using cubic splines and arc-length parameterization.

    Parameters
    ----------
    stroke : (N,2) ndarray
        Stroke coordinates normalized to [0,1].
    n_points : int
        Number of output samples.

    Returns
    -------
    (n_points,2) ndarray
        Resampled stroke with coordinates clipped to [0,1].
    """
    stroke = np.asarray(stroke, dtype=float)

    if len(stroke) == 0:
        return np.empty((0, 2))

    if len(stroke) == 1:
        return np.repeat(stroke, n_points, axis=0)

    # Arc-length parameter
    d = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(d)])

    total_length = s[-1]

    if total_length == 0:
        return np.repeat(stroke[:1], n_points, axis=0)

    # Remove duplicate arc-length values
    keep = np.concatenate([[True], np.diff(s) > 1e-12])
    s = s[keep]
    pts = stroke[keep]

    # If too few distinct points remain
    if len(pts) == 1:
        return np.repeat(pts, n_points, axis=0)

    # Choose spline order
    bc = "natural"

    sx = CubicSpline(s, pts[:, 0], bc_type=bc)
    sy = CubicSpline(s, pts[:, 1], bc_type=bc)

    s_new = np.linspace(0, total_length, n_points)

    out = np.column_stack([sx(s_new), sy(s_new)])

    # Cubic splines can overshoot slightly
    return np.clip(out, 0.0, 1.0)
```

You can then resample an entire kanji:

```python
def resample_kanji(strokes, n_points=32):
    return [resample_stroke(stroke, n_points) for stroke in strokes]
```

## Length-dependent number of samples

If you prefer the number of samples to depend on stroke length, compute the lengths first and allocate samples proportionally.

```python
def resample_kanji_by_length(
    strokes,
    total_points=256,
    min_points=8,
):
    """
    Allocate samples proportional to stroke length.
    """

    lengths = []
    for stroke in strokes:
        if len(stroke) < 2:
            lengths.append(0.0)
        else:
            lengths.append(
                np.sum(np.linalg.norm(np.diff(stroke, axis=0), axis=1))
            )

    lengths = np.asarray(lengths)
    total_length = lengths.sum()

    if total_length == 0:
        counts = [min_points] * len(strokes)
    else:
        counts = np.maximum(
            min_points,
            np.round(total_points * lengths / total_length).astype(int),
        )

    return [
        resample_stroke(stroke, int(n))
        for stroke, n in zip(strokes, counts)
    ]
```

### A small improvement

For handwriting data, `CubicSpline` may introduce slight oscillations near sharp corners. A shape-preserving spline such as `PchipInterpolator` often produces better-looking stroke resampling while still being cubic:

```python
from scipy.interpolate import PchipInterpolator

sx = PchipInterpolator(s, pts[:, 0])
sy = PchipInterpolator(s, pts[:, 1])
```

It avoids overshooting at corners and is commonly preferred for pen trajectories, although if you specifically want smooth second derivatives, `CubicSpline` is the appropriate choice.

> Ok. Amend the code to use PchipInterpolator.
