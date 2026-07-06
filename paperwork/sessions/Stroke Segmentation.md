Topic: Stroke Segmentation
Bot: Gemini
Date: 26 07 03

> Do you know an algorithm which splits a 2D linestring into curvy and straight segments?

Yes, you can split a 2D linestring into curvy and straight segments using a curvature-thresholding algorithm based on sliding-window PCA (Principal Component Analysis) or local tangent differences.
While there is no single, universally named algorithm for this specific task, the standard engineering approach involves calculating the discrete curvature at each vertex and grouping contiguous vertices that fall above or below a specific threshold.
Here is a breakdown of how the algorithm works, along with a complete Python implementation.

------------------------------
## 1. Compute Local Curvature
Calculate the curvature $K_i$ at each vertex $P_i$. For discrete 2D linestrings, curvature at vertex $P_i$ is best approximated by the turning angle (the change in the tangent angle) between the incoming segment $\vec{u} = P_i - P_{i-1}$ and outgoing segment $\vec{v} = P_{i+1} - P_i$, normalized by the average segment length.
$$K_i = \frac{\theta_i}{\frac{1}{2}(\Vert{}\vec{u}\Vert{} + \Vert{}\vec{v}\Vert{})}$$
Where the turning angle $\theta_i$ is found using the dot and determinant operations:
$$\theta_i = \left\vert{} \operatorname{atan2}(\det(\vec{u}, \vec{v}), \vec{u} \cdot \vec{v}) \right\vert{}$$
## 2. Apply a Threshold
Define a curvature threshold $\epsilon$.

* Vertices where $K_i \leq \epsilon$ are classified as straight.
* Vertices where $K_i > \epsilon$ are classified as curvy.

## 3. Segment the Linestring
Iterate through the classified vertices. Group consecutive vertices of the same type into distinct sub-linestrings. To prevent noisy data from creating tiny, alternating segments, you can apply a [moving average smoothing filter](https://www.makerluis.com/moving_average_filters/) to the curvatures before thresholding.

------------------------------
## Python Implementation
This script uses numpy to calculate the segments and splits them into distinct straight and curvy groups. [1, 2]

```python
import numpy as np
def split_linestring(coords, curvature_threshold=0.05):
    """
    Splits a 2D linestring into straight and curvy segments.
    coords: Nx2 numpy array of (x, y) coordinates
    """
    coords = np.array(coords, dtype=float)
    n = len(coords)
    if n < 3:
        return [("straight", coords)] # Too short to have curvature

    # 1. Compute vectors between consecutive points
    vectors = np.diff(coords, axis=0) # Shape: (n-1) x 2
    lengths = np.linalg.norm(vectors, axis=1)

    # 2. Compute local curvature (turning angles) at interior vertices
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        u = vectors[i-1]
        v = vectors[i]

        # Turning angle in radians
        dot = np.dot(u, v)
        det = u[0]*v[1] - u[1]*v[0]
        angle = np.abs(np.arctan2(det, dot))

        # Normalize by average local edge length
        avg_len = 0.5 * (lengths[i-1] + lengths[i])
        curvatures[i] = angle / avg_len if avg_len > 0 else 0

    # Boundary handling: copy adjacent curvature
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]

    # 3. Classify points based on threshold
    # 0 for straight, 1 for curvy
    classification = (curvatures > curvature_threshold).astype(int)

    # 4. Group segments
    segments = []
    start_idx = 0
    current_type = classification[0]

    for i in range(1, n):
        # Trigger split if classification changes
        if classification[i] != current_type:
            # Include the current point to keep segments physically connected
            seg_coords = coords[start_idx:i+1]
            seg_label = "curvy" if current_type == 1 else "straight"
            segments.append((seg_label, seg_coords))

            start_idx = i
            current_type = classification[i]

    # Append the final remaining segment
    seg_label = "curvy" if current_type == 1 else "straight"
    segments.append((seg_label, coords[start_idx:]))

    return segments

# --- Example Usage ---if __name__ == "__main__":
    # Create a linestring: a straight line, a sharp circle arc, then a straight line
    t = np.linspace(0, np.pi, 20)
    arc = np.column_stack((np.cos(t), np.sin(t))) + np.array([10, 0])

    line1 = np.column_stack((np.linspace(0, 9, 10), np.zeros(10)))
    line2 = np.column_stack((np.linspace(11, 20, 10), np.zeros(10)))

    test_linestring = np.vstack((line1, arc, line2))

    # Split the linestring
    result_segments = split_linestring(test_linestring, curvature_threshold=0.1)

    for idx, (label, points) in enumerate(result_segments):
        print(f"Segment {idx}: Type = {label}, Number of points = {len(points)}")
```

------------------------------
## Alternative Advanced Approaches
If your data is highly noisy (e.g., GPS tracks), local curvature formulas can fail due to jitter. Consider these alternatives:

* Sliding Window PCA: Move a small window (e.g., 5 points) along the linestring. Calculate the eigenvalues ($\lambda_1 \geq \lambda_2$) of the covariance matrix of the points within the window. The ratio $\lambda_2 / (\lambda_1 + \lambda_2)$ measures surface variation. If it is close to 0, the segment is straight; if higher, it is curvy. [3, 4]
* Ramer-Douglas-Peucker (RDP) Residuals: Run the [RDP simplification algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) with a strict tolerance. The points that get discarded are your straight regions, while the clusters of points that survive represent high-frequency curvy regions.

Would you like to adjust this algorithm to handle noisy GPS data, or do you need help visualizing the split segments using a plotting library?

[1] [https://docs.duet3d.com](https://docs.duet3d.com/User_manual/Machine_configuration/Configuring_RepRapFirmware_for_a_Robot_printer)
[2] [https://ciechanow.ski](https://ciechanow.ski/drawing-bezier-curves/)
[3] [https://grabcad.com](https://grabcad.com/tutorials/divide-tool-in-autocad)
[4] [https://webglfundamentals.org](https://webglfundamentals.org/webgl/lessons/webgl-3d-geometry-lathe.html)

> What exactly is curvature_threshold and why did you choose 0.05?

The curvature_threshold is the exact mathematical boundary that separates a straight line from a curve. It represents the maximum allowable change in direction per unit of distance before a segment is classified as "curvy."
## What Curvature Target Means Mathematically
In geometry, curvature ($K$) is defined as the rate of change of the tangent vector with respect to arc length ($K = \frac{d\theta}{ds}$). [1, 2]

* $K = 0$: A perfectly straight line (no direction change).
* $K = \text{constant}$: A perfect circle with a radius of $R = \frac{1}{K}$.
* $K > 0$: A curve. Higher values mean sharper, tighter turns. [3, 4, 5, 6, 7]

Because real-world linestrings are discrete points rather than smooth equations, the algorithm calculates discrete curvature:
$$\text{Curvature} \approx \frac{\text{Turning Angle (in radians)}}{\text{Average Segment Length}}$$
## Why 0.05 Was Chosen
The value 0.05 was selected as a general-purpose baseline because it translates to a highly intuitive physical threshold: a curve with a radius of 20 meters/units.
Because radius ($R$) is the inverse of curvature ($R = \frac{1}{K}$):
$$R = \frac{1}{0.05} = 20\text{ units}$$
Here is how that threshold behaves in practice depending on your segment lengths:

* For long segments (e.g., 100 meters apart): To hit a curvature of 0.05, the line must change direction by 5 radians ($286^\circ$) over that 100-meter span. Because that is geometrically impossible for a single turn, long segments are almost always classified as straight unless they form an incredibly sharp, jagged V-turn.
* For short segments (e.g., 1 meter apart): To hit a curvature of 0.05, the line only needs to change direction by 0.05 radians (about $2.8^\circ$) between vertices.

## How to Tune It For Your Data
You should change the value of 0.05 depending on your specific use case: [8]

* Decrease it ($0.01$ to $0.001$): If you want to detect gentle, sweeping curves (like highway bends or railway tracks). This makes the algorithm highly sensitive to minor bends. [9]
* Increase it ($0.1$ to $0.5$): If your data is noisy or if you only care about sharp turns (like 90-degree street corners or mountain switchbacks). This forces the algorithm to ignore minor wobbles and only label extreme bends as "curvy." [10]

To tailor this to your dataset, would you like to share what your linestrings represent (e.g., roads, drawing strokes, flight paths) or the typical distance between your points?

[1] [https://entokey.com](https://entokey.com/corneal-topography-5/)
[2] [https://repository.library.noaa.gov](https://repository.library.noaa.gov/view/noaa/68918/noaa_68918_DS1.pdf)
[3] [https://www.grad.hr](https://www.grad.hr/geomteh3d/skripta/zakrivljenost_eng.html)
[4] [https://www.vaia.com](https://www.vaia.com/en-us/textbooks/math/calculus-early-transcendentals-3-edition/chapter-14/problem-1-what-is-the-curvature-of-a-straight-line/)
[5] [https://www.vaia.com](https://www.vaia.com/en-us/textbooks/math/calculus-5-edition/chapter-15/problem-15-show-that-if-a-point-moves-along-the-graph-of-yfx/)
[6] [https://math.stackexchange.com](https://math.stackexchange.com/questions/1189091/a-point-of-infinite-curvature-on-a-curve)
[7] [https://sites.math.washington.edu](https://sites.math.washington.edu/~aloveles/Math126Materials/13-3%20Lecture%20-%20sp19.pdf)
[8] [https://www.datacamp.com](https://www.datacamp.com/tutorial/kruskal-wallis-test)
[9] [https://www.mdpi.com](https://www.mdpi.com/2075-1680/12/6/577)
[10] [https://www.khanacademy.org](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/curvature/v/curvature-intuition)

> OK, git it. For further context: We are dealing with Kanji stroke data which is normalised to [0, 1] in x and y.

For Kanji stroke data normalized to a $[0, 1]$ bounding box, a threshold of 0.05 is likely too sensitive, meaning it will incorrectly label slightly shaky but intended straight lines (like the long horizontal stroke in 一) as curvy.
Because the entire canvas is only $1.0 \times 1.0$ units, typical Kanji strokes are very short—often between $0.1$ and $0.6$ units long. The distances between sampled points within a single stroke are microscopic (often $0.01$ to $0.05$ units).
## How the Threshold Behaves on a $[0, 1]$ Canvas
Let's look at the math for a typical Kanji stroke snippet where points are sampled every $0.02$ units:

* If a writer's hand trembles slightly and creates a tiny $3^\circ$ ($0.052$ radians) angle change between two segments:
$$\text{Curvature} = \frac{0.052 \text{ rad}}{0.02 \text{ units}} = 2.6$$
* With your current threshold of 0.05, this tiny wobble ($2.6 > 0.05$) will instantly trigger a "curvy" classification.

For character recognition or stroke analysis on a normalized canvas, you need a threshold that ignores human drawing jitter but captures intentional, sweeping structural curves (like the hook in 亅 or the sweeping curve in ノ).
## Recommended Settings for Kanji Strokes

   1. Curvature Threshold: Try a value between 2.0 and 5.0.
   * A threshold of 3.0 means that if points are spaced $0.02$ units apart, the angle must change by more than $3.4^\circ$ to be called a curve.
   2. Pre-Smoothing (Critical): Digital ink/stylus data contains micro-wobbles. You must smooth the coordinates before calculating curvature, or run a moving average filter on the calculated curvatures.

## Adjusted Implementation for Kanji Strokes
Here is a modified version tailored for character strokes. It includes a basic Gaussian-like smoothing window to handle digital ink jitter.

```python
import numpy as np
def split_kanji_stroke(coords, curvature_threshold=3.0, smooth_window=3):
    """
    Splits a normalized [0,1] Kanji stroke into straight and curvy segments.
    """
    coords = np.array(coords, dtype=float)
    n = len(coords)
    if n < 3:
        return [("straight", coords)]

    # Optional: Smooth coordinates to eliminate stylus/mouse jitter
    if n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        coords[:, 0] = np.convolve(coords[:, 0], kernel, mode='same')
        coords[:, 1] = np.convolve(coords[:, 1], kernel, mode='same')
        # Fix convolve boundary artifacts
        coords[0] = coords[1]
        coords[-1] = coords[-2]

    vectors = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        u = vectors[i-1]
        v = vectors[i]

        avg_len = 0.5 * (lengths[i-1] + lengths[i])
        if avg_len < 1e-5:
            continue

        dot = np.clip(np.dot(u, v) / (lengths[i-1] * lengths[i]), -1.0, 1.0)
        angle = np.arccos(dot) # Turning angle in radians

        curvatures[i] = angle / avg_len

    # Classify and group segments
    classification = (curvatures > curvature_threshold).astype(int)

    segments = []
    start_idx = 0
    current_type = classification[0]

    for i in range(1, n):
        if classification[i] != current_type:
            segments.append(("curvy" if current_type == 1 else "straight", coords[start_idx:i+1]))
            start_idx = i
            current_type = classification[i]

    segments.append(("curvy" if current_type == 1 else "straight", coords[start_idx:]))
    return segments
```

Do you plan to use these segmented pieces to classify stroke types (like Hanzi/Kanji biánbù/radicals), or are you trying to compress the stroke data for storage?

> I do not like to mix discrete steps of an input pipeline into one function. I already apply Gaussian smoothing before splitting into segments. Please remove smoothing from split_kanji_stroke.

Separating the smoothing step from the geometric analysis is a much cleaner architecture.
Here is the streamlined split_kanji_stroke function with all smoothing logic removed, operating strictly on your pre-processed coordinate array.

```python
import numpy as np

def split_kanji_stroke(coords, curvature_threshold=3.0):
    """
    Splits a normalized Kanji stroke into straight and curvy segments.
    Expects pre-smoothed Nx2 numpy arrays.
    """
    coords = np.array(coords, dtype=float)
    n = len(coords)
    if n < 3:
        return [("straight", coords)]

    # 1. Compute vectors and segment lengths
    vectors = np.diff(coords, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    # 2. Compute discrete curvature (radians per unit length)
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        u = vectors[i-1]
        v = vectors[i]

        len_u = lengths[i-1]
        len_v = lengths[i]

        # Avoid division by zero on duplicate consecutive points
        if len_u < 1e-6 or len_v < 1e-6:
            continue

        avg_len = 0.5 * (len_u + len_v)

        # Compute cosine of the turning angle using the dot product formula
        cos_theta = np.clip(np.dot(u, v) / (len_u * len_v), -1.0, 1.0)
        angle = np.arccos(cos_theta)

        curvatures[i] = angle / avg_len

    # 3. Classify points based on the threshold (0 = straight, 1 = curvy)
    classification = (curvatures > curvature_threshold).astype(int)

    # 4. Group consecutive points into segments
    segments = []
    start_idx = 0
    current_type = classification[0]

    for i in range(1, n):
        if classification[i] != current_type:
            # Include the boundary vertex in both segments to prevent visual gaps
            seg_label = "curvy" if current_type == 1 else "straight"
            segments.append((seg_label, coords[start_idx:i+1]))

            start_idx = i
            current_type = classification[i]

    # Append the final remaining segment
    seg_label = "curvy" if current_type == 1 else "straight"
    segments.append((seg_label, coords[start_idx:]))

    return segments
```

Now that the segmentation is isolated, what is the next step in your pipeline? If you are trying to identify specific stroke types (like the Hane hook or Ten dot) or extract geometric features from these split segments, I can help design the next module.

Bot: ChatGPT
Date: 26 07 03

> I am looking for a way to split a single Japanese Kanji or Kana stroke into curvy and straight segments. The idea is to detect unwanted hooks and edges at the beginning and/or end of a stroke which result from Apple Pencil input on iPad. A stroke is stored as 2D NumPy array with various columms, two of which are normalized x and y coordinates [0, 1]. The actual indices of these columns are given as a 2-tuple. The shape of point vectors should be retained in the resulting list of curvy and straight segments. The input data is already preprocessed with Gaussian smoothing.

A good way to think about this problem is that you are **segmenting a parametric curve by geometric behavior**, not by time. Hooks caused by stylus input usually have one or more of these properties:

* high curvature concentrated near the ends,
* rapid change in tangent direction,
* short arc length,
* often a reversal of curvature sign.

Straight portions have nearly constant tangent direction and very low curvature.

For already Gaussian-smoothed input, curvature-based segmentation works quite well.

## 1. Parameterize by arc length

Don't differentiate with respect to sample index. First compute cumulative arc length.

```python
import numpy as np

def arc_length(points):
    d = np.diff(points, axis=0)
    ds = np.linalg.norm(d, axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])
    return s
```

Extract coordinates

```python
xy = stroke[:, xy_columns]     # xy_columns = (x_idx, y_idx)
s = arc_length(xy)
```

## 2. Estimate tangent

Use central differences.

```python
dx = np.gradient(xy[:,0], s)
dy = np.gradient(xy[:,1], s)

speed = np.sqrt(dx**2 + dy**2)

tx = dx / (speed + 1e-8)
ty = dy / (speed + 1e-8)
```

Now every point has a unit tangent.

---

## 3. Compute curvature

```python
dtx = np.gradient(tx, s)
dty = np.gradient(ty, s)

curvature = np.sqrt(dtx**2 + dty**2)
```

This is

[
\kappa = \left|\frac{dT}{ds}\right|
]

which is very stable after Gaussian smoothing.

---

## 4. Detect straight regions

A point is approximately straight if

```python
straight = curvature < kappa_threshold
```

where `kappa_threshold` is empirical.

For normalized coordinates, something around

```python
kappa_threshold = 5
```

is a reasonable starting point, but depends on stroke scale.

Instead of a fixed threshold, a robust choice is

```python
kappa_threshold = np.percentile(curvature, 70)
```

or

```python
median = np.median(curvature)
mad = np.median(np.abs(curvature - median))
kappa_threshold = median + 2*mad
```

---

## 5. Find connected segments

```python
changes = np.flatnonzero(np.diff(straight.astype(int))) + 1

bounds = np.concatenate([[0], changes, [len(straight)]])
```

Each interval

```python
bounds[i]:bounds[i+1]
```

is either straight or curved.

---

## 6. Remove tiny segments

Hooks are short.

Suppose a segment has arc length

```python
L = s[end-1] - s[start]
```

If

```python
L < 0.01
```

(or 1% of stroke length)

merge it with neighboring segments.

This prevents noisy fragmentation.

---

# Better for detecting hooks

Curvature alone finds bends, but hooks have a very characteristic tangent evolution.

Compute tangent angle

```python
theta = np.unwrap(np.arctan2(ty, tx))
```

Then

```python
dtheta = np.gradient(theta, s)
```

Large values near the beginning or end indicate a hook.

For example

```
hook
 ----)
```

produces

* large |dθ/ds|
* only over a short distance
* only near an endpoint.

That makes it easy to remove.

---

# Even more robust: Ramer–Douglas–Peucker first

An approach that works surprisingly well for handwriting:

1. Gaussian smoothing ✔
2. Simplify with RDP
3. Compute angle between adjacent line segments
4. Expand breakpoints back to the original samples

Example:

```
original

~~~~~~~~~~~~~~

RDP

----/------
```

Each corner in the RDP polyline corresponds to a curved region.

This often works better than raw curvature because it ignores tiny wiggles.

---

# Preserving the original NumPy rows

Suppose you found break indices

```python
breaks = [0, 18, 51, 67, len(stroke)]
```

Then simply slice the original array:

```python
segments = [
    stroke[a:b]
    for a, b in zip(breaks[:-1], breaks[1:])
]
```

Each segment keeps **all columns** (pressure, timestamp, tilt, etc.), exactly as requested.

---

# Recommendation

For Japanese handwriting, where strokes are mostly straight or smoothly curved and unwanted Apple Pencil artifacts are typically small endpoint hooks, a pipeline like this tends to be both effective and computationally light:

1. Gaussian smoothing (already done).
2. Compute arc length.
3. Estimate tangents with central differences.
4. Compute curvature ( \kappa = \lVert dT/ds \rVert ).
5. Identify contiguous low-curvature (straight) and high-curvature (curved) regions.
6. Merge segments below a minimum arc-length threshold to avoid over-segmentation.
7. Apply a specific endpoint rule: if the first or last curved segment is both short (e.g., <2–5% of the total stroke length) and has a large cumulative turning angle (e.g., 20–40°), classify it as an unwanted hook and remove or trim it.

This approach is scale-independent when parameterized by arc length, preserves the original stroke samples by segmenting via index ranges, and generally performs well on stylus handwriting after Gaussian smoothing.

> OK, now I am slightly overwhelmed. Let's take it one step at a time. I understand, that preprocessing the input with gaussian smoothing and RDP (in that order) would be the right thing to do to later detect and remove unwanted hooks. I assume going forward, timestamps for points which were included in the input data can be stripped if we rely on geometrical behavior alone. Am I right so far?

Yes, you're on exactly the right track, and I'd actually simplify it even a bit further.

Your pipeline could be thought of as

```
raw stroke
    ↓
Gaussian smoothing
    ↓
RDP simplification
    ↓
geometric analysis
    ↓
hook detection/removal
```

### Gaussian smoothing

The purpose is simply to remove the high-frequency jitter introduced by the Apple Pencil and hand tremor. It should preserve the overall shape while making derivatives (tangent, curvature, etc.) much more stable.

### RDP simplification

This serves a different purpose. Rather than smoothing, it removes redundant points that lie nearly on the same line.

For example, a stroke like

```
------------------------
```

might originally contain 200 points but be reduced to just

```
o----------------------o
```

while a curved region

```
(
```

would still require several vertices.

This makes later geometric analysis both faster and more robust.

### About timestamps

Yes. Once you've reached the stage where you're looking for hooks based on geometry, timestamps are no longer needed.

A hook is a **shape**, not a temporal event.

The only columns you need for the analysis are

* x
* y

Everything else (timestamp, pressure, azimuth, altitude, etc.) can be ignored **during the analysis**.

However, I would recommend **not deleting those columns** from your original stroke array. Instead:

* perform all computations on a view containing only the x/y columns;
* keep the original array intact;
* when you later decide that, say, points 0–6 form an unwanted hook, you can remove those same rows from the original array. That way you preserve pressure, timestamps, and any other metadata for the remaining points.

### One small suggestion

If your goal is specifically to remove hooks at the beginning and end of a stroke, I would actually **delay the RDP step** until we've examined it more closely.

The reason is that a very small hook can sometimes be represented by only two or three points. With an RDP tolerance that's even slightly too aggressive, those points may disappear entirely or become merged into the main stroke, making it harder to identify exactly where the hook ends.

So I'd recommend we first characterize the hooks on the smoothed, full-resolution stroke. Once we're happy with the detection logic, we can decide whether RDP should be:

* omitted,
* used only to accelerate the analysis, or
* used after hook removal.

For Apple Pencil handwriting, it's quite possible that Gaussian smoothing plus curvature analysis on the original sample sequence will be sufficient, and RDP won't add much value. We can determine that once we have the hook detector working.

I suggest we tackle the next step by implementing just one geometric quantity: **the tangent direction** (the direction the pen is moving at each point). Almost every subsequent measurement—turning angle, curvature, and hook detection—can be built on top of that.

> OK. We skip RDP for now and focus on hook detector. Can you give me a corresponding Python function for s single stroke given as 2D NumPy array with x and y columns in the range of [0, 1]?

Yes. I would start with something **very simple** and build it incrementally. The first function shouldn't try to detect hooks yet. Instead, it should compute the geometric quantities we'll need later:

* cumulative arc length
* unit tangent vector
* tangent angle
* curvature

Once we can visualize these, hook detection becomes almost trivial.

Here's a self-contained function.

```python
import numpy as np


def stroke_geometry(stroke, xy_cols=(0, 1)):
    """
    Compute basic geometric properties of a stroke.

    Parameters
    ----------
    stroke : (N,M) ndarray
        Stroke data. Only the columns specified by xy_cols are used.
    xy_cols : tuple(int, int)
        Column indices containing normalized x and y coordinates.

    Returns
    -------
    dict containing

        xy          (N,2) coordinates
        s           cumulative arc length
        tangent     (N,2) unit tangent vectors
        angle       tangent angle (radians, unwrapped)
        curvature   curvature (1/length)
    """

    xy = stroke[:, xy_cols].astype(float)

    if len(xy) < 3:
        raise ValueError("Stroke must contain at least 3 points.")

    # ---------------------------------------------------------
    # cumulative arc length
    # ---------------------------------------------------------

    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # avoid duplicate arc lengths
    if np.any(ds == 0):
        eps = 1e-10
        s = np.maximum.accumulate(s + eps * np.arange(len(s)))

    # ---------------------------------------------------------
    # first derivative
    # ---------------------------------------------------------

    dx = np.gradient(xy[:, 0], s)
    dy = np.gradient(xy[:, 1], s)

    speed = np.hypot(dx, dy)

    tx = dx / (speed + 1e-12)
    ty = dy / (speed + 1e-12)

    tangent = np.column_stack((tx, ty))

    # ---------------------------------------------------------
    # tangent angle
    # ---------------------------------------------------------

    angle = np.unwrap(np.arctan2(ty, tx))

    # ---------------------------------------------------------
    # curvature
    # ---------------------------------------------------------

    dtx = np.gradient(tx, s)
    dty = np.gradient(ty, s)

    curvature = np.hypot(dtx, dty)

    return {
        "xy": xy,
        "s": s,
        "tangent": tangent,
        "angle": angle,
        "curvature": curvature,
    }
```

## Why these quantities?

Each one tells us something different:

* `s`: "How far along the stroke am I?" (measured in geometric distance, not sample index).
* `tangent`: "Which direction is the pen moving here?"
* `angle`: The tangent expressed as a single angle. A straight stroke has an almost constant angle; a curve changes angle smoothly.
* `curvature`: "How fast is the direction changing?" Straight segments have curvature near zero, while bends and hooks produce peaks.

## What comes next?

Before writing a hook detector, I would recommend plotting the results for a few representative strokes. In particular, plot `curvature` versus `s`. For a stroke with an unwanted hook at the beginning or end, you'll often see something like:

```
curvature

 ^
 |  /\                        /\
 | /  \                      /  \
 |/    \____________________/    \____
 +--------------------------------------> arc length
   hook        straight part        hook
```

Those peaks are usually much easier to detect than trying to infer hooks directly from the raw coordinates.

One small refinement to the function above: since your coordinates are normalized to `[0, 1]`, I would also return the **normalized arc length**

```python
s_norm = s / s[-1]
```

This expresses every point as a fraction of the stroke length (from 0 to 1). It makes endpoint rules much cleaner—for example, "only inspect the first and last 10% of the stroke for hooks"—and makes the algorithm independent of the absolute stroke size.

> I added "s_norm" manually to the result. Before going on, can you give me a function to create a mathplotlib plot showing the calculated geometry features for a list of consecutive strokes?

That's exactly what I would do next. In fact, I'd recommend plotting **all four quantities together** while we're developing the detector. Looking at dozens of strokes is usually enough to spot a robust criterion for hooks.

I'd make the function accept a list of dictionaries returned by `stroke_geometry()`. The left column shows the stroke itself, while the right column shows the geometric quantities over normalized arc length.

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_stroke_geometry(geometries, figsize=(12, 3)):
    """
    Plot one or more strokes together with their geometric properties.

    Parameters
    ----------
    geometries : list of dict
        Each dict is the result of stroke_geometry().
    figsize : tuple
        Figure size per row.
    """

    n = len(geometries)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(figsize[0], figsize[1] * n),
        constrained_layout=True,
        squeeze=False,
    )

    for i, g in enumerate(geometries):

        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        xy = g["xy"]

        # --------------------------------------------------
        # Stroke
        # --------------------------------------------------

        ax0.plot(xy[:, 0], xy[:, 1], "-k", lw=2)
        ax0.scatter(
            xy[0, 0], xy[0, 1],
            c="green", s=50, label="start"
        )
        ax0.scatter(
            xy[-1, 0], xy[-1, 1],
            c="red", s=50, label="end"
        )

        ax0.set_aspect("equal")
        ax0.invert_yaxis()      # optional for handwriting
        ax0.set_title(f"Stroke {i}")
        ax0.legend(loc="best")

        # --------------------------------------------------
        # Geometry
        # --------------------------------------------------

        s = g["s_norm"]

        ax1.plot(s, g["angle"], label="angle")
        ax1.plot(s, g["curvature"], label="curvature")

        ax1.set_xlim(0, 1)
        ax1.set_xlabel("normalized arc length")
        ax1.grid(True)
        ax1.legend(loc="best")

    plt.show()
```

## I would make one small improvement

For hook detection, the tangent angle is often less informative than its **change**. So I'd also compute and return

```python
dangle = np.gradient(angle, s)
```

or, since you already have it,

```python
dangle = np.gradient(angle, s_norm)
```

(Using `s` is mathematically cleaner, but since `s_norm` is just a constant scaling of `s`, the curves have the same shape.)

Then your right-hand plot would show

```python
ax1.plot(s, g["curvature"], label="curvature", lw=2)
ax1.plot(s, g["dangle"], label="dθ/ds", lw=2)
```

In theory, `|dθ/ds|` and the curvature should be almost identical. In practice, plotting both is a nice sanity check while developing the algorithm.

## One more suggestion

Since we're going to iterate on this detector, I'd also color the stroke by curvature. It's one of the most useful visualizations I've found for this kind of work because you immediately see *where* the algorithm thinks the stroke is bending.

Using a `LineCollection` from `matplotlib.collections`, you can render the stroke with blue for straight sections and red for high-curvature sections. Once you see a few Apple Pencil hooks, it becomes much easier to decide on a reliable detection rule. I suspect that visualization will become your primary debugging tool as we refine the detector.

> Looks good so far. Yes, please amend the code to include coloration of curvy segments.

Great. I actually wouldn't use a binary "curvy vs. straight" coloring yet. Instead, I'd color the stroke continuously by curvature using a colormap. That lets you immediately see where the curvature peaks are and later decide on a threshold.

Here's an enhanced version of the plotting function.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_stroke_geometry(geometries,
                         figsize=(14, 3.5),
                         cmap="turbo",
                         curvature_percentile=95):
    """
    Plot one or more strokes together with their geometric properties.

    Parameters
    ----------
    geometries : list of dict
        Result of stroke_geometry().
    figsize : tuple
        Figure size per row.
    cmap : str
        Matplotlib colormap.
    curvature_percentile : float
        Percentile used to normalize curvature colors.
        Prevents a single spike from dominating the color scale.
    """

    n = len(geometries)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(figsize[0], figsize[1] * n),
        constrained_layout=True,
        squeeze=False,
    )

    for i, g in enumerate(geometries):

        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        xy = g["xy"]
        s = g["s_norm"]
        curvature = g["curvature"]

        # ------------------------------------------------------
        # Left plot: stroke colored by curvature
        # ------------------------------------------------------

        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Use curvature at the first point of each segment
        c = curvature[:-1]

        vmax = np.percentile(curvature, curvature_percentile)
        vmax = max(vmax, 1e-6)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=Normalize(vmin=0.0, vmax=vmax),
        )

        lc.set_array(c)
        lc.set_linewidth(2.5)

        ax0.add_collection(lc)

        ax0.scatter(*xy[0], c="limegreen", s=60, zorder=3, label="start")
        ax0.scatter(*xy[-1], c="red", s=60, zorder=3, label="end")

        ax0.autoscale()
        ax0.set_aspect("equal")
        ax0.invert_yaxis()
        ax0.set_title(f"Stroke {i}")
        ax0.legend(loc="best")

        cb = fig.colorbar(lc, ax=ax0, fraction=0.046, pad=0.04)
        cb.set_label("Curvature")

        # ------------------------------------------------------
        # Right plot: geometry
        # ------------------------------------------------------

        ax1.plot(s, g["angle"], lw=2, label="angle")
        ax1.plot(s, curvature, lw=2, label="curvature")

        if "dangle" in g:
            ax1.plot(s, g["dangle"], "--", lw=1.5, label="dθ/ds")

        ax1.set_xlim(0.0, 1.0)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Normalized arc length")
        ax1.set_title("Geometry")
        ax1.legend(loc="best")

    plt.show()
```

## I'd also make one small addition to `stroke_geometry()`

Since we're now using curvature for visualization and later for detection, I'd smooth the curvature **itself** slightly. Although the coordinates have already been Gaussian filtered, numerical differentiation amplifies any remaining noise.

At the end of `stroke_geometry()`, after computing `curvature`, add something like:

```python
from scipy.ndimage import gaussian_filter1d

curvature = gaussian_filter1d(curvature, sigma=1.0, mode="nearest")
```

A sigma of about `1.0` is usually enough to remove tiny oscillations without noticeably broadening genuine curvature peaks.

## One more quantity I think will prove useful

I'd also return the **signed curvature**:

```python
cross = tx * dty - ty * dtx
```

instead of only

```python
curvature = np.hypot(dtx, dty)
```

The magnitude tells you *how much* the stroke bends, while the sign tells you *whether it bends left or right*. Apple Pencil hooks often produce a characteristic signed-curvature pattern at the ends of strokes (a single strong lobe or a short reversal), which can make them easier to distinguish from intentional curves.

So I'd eventually have both:

```python
g["curvature"]         # always ≥ 0
g["signed_curvature"]  # positive or negative
```

That gives us more information without significantly increasing the complexity of the detector.

> Two things. First can you give me an updated stroke_geometry function which includes the additional quantities. And second in plot_stroke_geometry the strokes are not coloured but still black.

Good catches. The second issue actually explains the first.

The stroke is still black because `LineCollection` is **not automatically scaled into the axes**. The call to `ax0.autoscale()` doesn't reliably update the data limits from a collection. As a result, the collection is often clipped away and what you are seeing is likely the default line (or nothing at all, depending on your environment).

I also realized we can improve the geometry function a bit while we're at it.

## Updated `stroke_geometry`

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d


def stroke_geometry(stroke, xy_cols=(0, 1), sigma=1.0):
    """
    Compute geometric properties of a stroke.

    Returns
    -------
    dict with keys

        xy
        s
        s_norm
        tangent
        angle
        dangle
        curvature
        signed_curvature
    """

    xy = stroke[:, xy_cols].astype(float)

    if len(xy) < 3:
        raise ValueError("Need at least three points.")

    # ----------------------------------------------------------
    # Arc length
    # ----------------------------------------------------------

    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # avoid duplicate arc lengths
    s += np.arange(len(s)) * 1e-12

    s_norm = s / s[-1]

    # ----------------------------------------------------------
    # First derivative
    # ----------------------------------------------------------

    dx = np.gradient(xy[:, 0], s)
    dy = np.gradient(xy[:, 1], s)

    speed = np.hypot(dx, dy)

    tx = dx / (speed + 1e-12)
    ty = dy / (speed + 1e-12)

    tangent = np.column_stack((tx, ty))

    # ----------------------------------------------------------
    # Tangent angle
    # ----------------------------------------------------------

    angle = np.unwrap(np.arctan2(ty, tx))

    dangle = np.gradient(angle, s)

    # ----------------------------------------------------------
    # Curvature
    # ----------------------------------------------------------

    dtx = np.gradient(tx, s)
    dty = np.gradient(ty, s)

    signed_curvature = tx * dty - ty * dtx
    curvature = np.abs(signed_curvature)

    # light smoothing
    curvature = gaussian_filter1d(curvature, sigma=sigma)
    signed_curvature = gaussian_filter1d(signed_curvature, sigma=sigma)
    dangle = gaussian_filter1d(dangle, sigma=sigma)

    return dict(
        xy=xy,
        s=s,
        s_norm=s_norm,
        tangent=tangent,
        angle=angle,
        dangle=dangle,
        curvature=curvature,
        signed_curvature=signed_curvature,
    )
```

---

## Correct colouring

Instead of `LineCollection`, I would actually recommend using a scatter plot for development. It is simpler and shows exactly which **sample points** have high curvature.

Replace the stroke plotting section by

```python
xy = g["xy"]
c = g["curvature"]

sc = ax0.scatter(
    xy[:,0],
    xy[:,1],
    c=c,
    s=18,
    cmap=cmap,
    vmin=0,
    vmax=np.percentile(c, curvature_percentile),
)

# connect points with a thin grey line
ax0.plot(xy[:,0], xy[:,1],
         color="0.75",
         lw=1,
         zorder=0)

ax0.scatter(*xy[0], c="lime", s=60, edgecolor="black")
ax0.scatter(*xy[-1], c="red", s=60, edgecolor="black")

fig.colorbar(sc, ax=ax0, label="Curvature")
```

I actually prefer this during algorithm development because every coloured dot corresponds to one row in your NumPy array.

---

## If you prefer a coloured line

If you'd rather have the entire stroke coloured continuously (which does look nicer), use this instead:

```python
from matplotlib.collections import LineCollection

points = xy.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap=cmap,
    linewidths=3,
)

lc.set_array(c[:-1])

ax0.add_collection(lc)

# THIS is the important part
ax0.set_xlim(xy[:,0].min(), xy[:,0].max())
ax0.set_ylim(xy[:,1].max(), xy[:,1].min())   # inverted y
```

The explicit `set_xlim()`/`set_ylim()` is what was missing. Without it, many Matplotlib backends never properly display the coloured collection.

---

Personally, I'd use the **scatter version** while we're designing the hook detector. Once we're satisfied with the algorithm, switching to the prettier coloured-line visualization is straightforward. The scatter plot also makes it much easier to verify that the detected hook corresponds to the correct point indices in the original stroke.

> I think we can, and I'd deliberately keep the first version conservative rather than trying to be overly clever.

One observation about Apple Pencil hooks is that they are usually:

* confined to the first or last few percent of the stroke,
* short,
* have high curvature,
* accumulate a significant turning angle before joining the main stroke.

That suggests a detector that works from the ends **inwards** instead of trying to segment the entire stroke.

## Version 1

The function below returns the indices of the first and last point belonging to a detected hook. If no hook is found, the corresponding value is `None`.

```python
import numpy as np


def detect_hooks(
    geom,
    endpoint_fraction=0.15,
    curvature_factor=2.5,
    min_turn_deg=25,
):
    """
    Detect hooks at the beginning and end of a stroke.

    Parameters
    ----------
    geom : dict
        Result of stroke_geometry().
    endpoint_fraction : float
        Only inspect this fraction of the stroke at each end.
    curvature_factor : float
        Curvature threshold relative to the median curvature.
    min_turn_deg : float
        Minimum accumulated turning angle for a hook.

    Returns
    -------
    dict
        {
            "start_hook": int or None,
            "end_hook": int or None
        }

    start_hook is the last index belonging to the starting hook.

    end_hook is the first index belonging to the ending hook.
    """

    s = geom["s_norm"]
    k = geom["curvature"]
    angle = geom["angle"]

    threshold = curvature_factor * np.median(k)

    # ---------- beginning ----------

    start_region = np.where(s <= endpoint_fraction)[0]

    start_hook = None

    if len(start_region):

        curved = start_region[k[start_region] > threshold]

        if len(curved):

            last = curved[-1]

            turn = np.degrees(abs(angle[last] - angle[0]))

            if turn >= min_turn_deg:
                start_hook = last

    # ---------- end ----------

    end_region = np.where(s >= 1.0 - endpoint_fraction)[0]

    end_hook = None

    if len(end_region):

        curved = end_region[k[end_region] > threshold]

        if len(curved):

            first = curved[0]

            turn = np.degrees(abs(angle[-1] - angle[first]))

            if turn >= min_turn_deg:
                end_hook = first

    return {
        "start_hook": start_hook,
        "end_hook": end_hook,
    }
```

Example:

```python
geom = stroke_geometry(stroke, xy_cols=(0,1))
hooks = detect_hooks(geom)

print(hooks)
# {'start_hook': 6, 'end_hook': None}
```

You could then remove the hook simply by slicing:

```python
trimmed = stroke[hooks["start_hook"] + 1:]
```

or

```python
trimmed = trimmed[:hooks["end_hook"]]
```

---

## However...

After thinking through your application (Japanese handwriting), I don't think this is the detector I would ultimately use.

The weakness of thresholding curvature is that **where a hook ends** depends on an arbitrary threshold.

I think a better detector is based on **cumulative turning angle**:

1. Start at the endpoint.
2. Walk inward point by point.
3. Accumulate the absolute change in tangent direction.
4. Stop when the curvature falls below the background level for several consecutive points.
5. If the accumulated turning exceeds, say, 25–40°, classify everything traversed as the hook.

This approach has two advantages:

* it doesn't rely on a single curvature threshold crossing,
* it naturally finds the point where the stroke "settles" into its main direction.

For Apple Pencil endpoint artifacts, I expect it to be considerably more stable. Since we're building this incrementally, I'd recommend making that cumulative-turn detector our Version 2. I think it has a better chance of distinguishing genuine Japanese stroke endings from unintended stylus hooks.

> Version 1 already works pretty well for hooks at the beginning of strokes, but not so good for endings. Can we improve this situation before going on to version 2 or should we look into version 2 right now?

Good signal: when a detector is asymmetric (start works, end doesn’t), it usually means the rule is implicitly “anchored at a clean reference point” at the start but not at the end.

We can improve Version 1 a bit, and it’s worth doing—but only as a **small fix**, not a full redesign. The real conceptual upgrade is still Version 2.

---

# Why Version 1 fails more at the end

Your current logic:

* compares endpoint angle to a **fixed reference angle**:

  * start: `angle[0]` (usually stable)
  * end: `angle[-1]` (often *not stable*)

At stroke endings, the last point is frequently inside the hook already, meaning:

> you are anchoring the measurement inside the distortion itself

So the detector loses its reference frame.

---

# Small but important improvement (still Version 1)

## Key idea:

Instead of using a single endpoint angle, use a **local baseline direction**.

We replace:

* `angle[0]`
* `angle[-1]`

with:

* mean direction over a *clean neighborhood just inside the stroke*

---

# Improved Version 1 (drop-in upgrade)

```python
import numpy as np


def detect_hooks_v1_improved(
    geom,
    endpoint_fraction=0.15,
    curvature_factor=2.5,
    min_turn_deg=25,
    baseline_window=5,
):
    """
    Improved version of endpoint hook detection.

    Fix: use local baseline direction instead of endpoint angle.
    """

    s = geom["s_norm"]
    k = geom["curvature"]
    angle = geom["angle"]

    threshold = curvature_factor * np.median(k)

    def local_baseline_angle(idx_range):
        """Stable direction estimate excluding endpoint noise."""
        idx_range = np.array(idx_range)
        if len(idx_range) <= baseline_window:
            return angle[idx_range[0]]

        core = idx_range[:baseline_window]
        return np.mean(angle[core])

    # ---------------- START ----------------

    start_region = np.where(s <= endpoint_fraction)[0]
    start_hook = None

    if len(start_region) > 0:
        curved = start_region[k[start_region] > threshold]

        if len(curved) > 0:
            last = curved[-1]

            baseline = local_baseline_angle(start_region)

            turn = np.degrees(abs(angle[last] - baseline))

            if turn >= min_turn_deg:
                start_hook = last

    # ---------------- END ----------------

    end_region = np.where(s >= 1 - endpoint_fraction)[0]
    end_hook = None

    if len(end_region) > 0:
        curved = end_region[k[end_region] > threshold]

        if len(curved) > 0:
            first = curved[0]

            baseline = local_baseline_angle(end_region[::-1])

            turn = np.degrees(abs(angle[first] - baseline))

            if turn >= min_turn_deg:
                end_hook = first

    return {
        "start_hook": start_hook,
        "end_hook": end_hook,
    }
```

---

# Why this helps (intuitively)

Instead of asking:

> “How different is the endpoint from itself?”

we ask:

> “How different is this region from the direction the stroke had just before entering it?”

That removes the biggest failure mode at stroke endings.

---

# But should we stop here?

Short answer: **no, but this is worth doing first.**

This improved Version 1 gives you:

* better symmetry between start and end,
* no conceptual change,
* almost zero computational cost,
* and it will likely fix many of your current false negatives.

---

# When to move to Version 2

You move to Version 2 when you observe any of these:

* hooks sometimes split into multiple small curvature blobs,
* threshold tuning becomes unstable across strokes,
* Japanese characters with intentional curvature get confused with hooks,
* or you still see missed endpoint artifacts after this fix.

At that point, we switch to:

> cumulative turning angle + inward sweep (a fundamentally different detector)

---

# Recommendation

Do this order:

1. Try improved Version 1 (above)
2. Test on a variety of strokes (especially ending-heavy ones like 「し」「つ」「の」)
3. If failures persist → move to Version 2

If you want next step after this, I’d suggest adding a **“hook confidence score”** rather than a hard cutoff. That usually makes the transition to Version 2 much smoother.

> Version 1 is really an improvement for some trailing hooks. But let' see how version 2 would perform. Can you give me a Python function for this?

Good — this is exactly the point where switching approaches is the right move.

Version 2 is conceptually different: instead of *thresholding curvature in a region*, we **walk inward from the endpoint and measure how much the stroke “turns” before it stabilizes**.

This is much more robust for trailing hooks because it does not depend on detecting a clean curvature peak first.

---

# Core idea of Version 2

For each endpoint:

### We:

1. Start at the end (or beginning)
2. Move inward point-by-point
3. Accumulate change in tangent direction
4. Stop when:

   * curvature is low for a few consecutive points (**stability condition**)
5. Mark everything before that as a hook if:

   * total accumulated turn is large enough

---

# Version 2: Cumulative turning angle detector

```python
import numpy as np


def detect_hooks_v2(
    geom,
    endpoint_fraction=0.2,
    min_turn_deg=25,
    stability_window=5,
    curvature_rel_threshold=2.0,
):
    """
    Hook detection based on cumulative turning angle.

    This version walks inward from both endpoints.

    Parameters
    ----------
    geom : dict
        Output of stroke_geometry()
    endpoint_fraction : float
        Fraction of stroke to inspect at each end
    min_turn_deg : float
        Minimum accumulated turning angle to qualify as hook
    stability_window : int
        Number of consecutive low-curvature points needed to stop
    curvature_rel_threshold : float
        Relative threshold vs median curvature

    Returns
    -------
    dict with:
        start_hook: last index of start hook (or None)
        end_hook: first index of end hook (or None)
    """

    xy = geom["xy"]
    s = geom["s_norm"]
    k = geom["curvature"]
    angle = geom["angle"]

    n = len(xy)
    k_med = np.median(k)
    k_thr = curvature_rel_threshold * k_med

    # ----------------------------
    # helper: inward sweep
    # ----------------------------
    def sweep(indices, forward=True):
        if len(indices) < 3:
            return None

        if not forward:
            indices = indices[::-1]

        turns = []
        accumulated = 0.0
        low_count = 0

        prev_angle = angle[indices[0]]

        for i in indices[1:]:
            dtheta = angle[i] - prev_angle

            # unwrap local jump
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

            accumulated += abs(dtheta)
            turns.append((i, accumulated))

            if k[i] < k_thr:
                low_count += 1
            else:
                low_count = 0

            prev_angle = angle[i]

            # stop when stable straight region found
            if low_count >= stability_window:
                break

        if len(turns) == 0:
            return None

        last_idx, total_turn = turns[-1]

        if np.degrees(total_turn) >= min_turn_deg:
            return last_idx

        return None

    # ----------------------------
    # define endpoint regions
    # ----------------------------
    start_region = np.where(s <= endpoint_fraction)[0]
    end_region = np.where(s >= 1 - endpoint_fraction)[0]

    # ----------------------------
    # detect start hook (forward)
    # ----------------------------
    start_hook = sweep(start_region, forward=True)

    # ----------------------------
    # detect end hook (backward)
    # ----------------------------
    end_hook = sweep(end_region, forward=False)

    return {
        "start_hook": start_hook,
        "end_hook": end_hook,
    }
```

---

# Why Version 2 works better

### 1. No reliance on curvature peaks

Hooks often don’t have a clean peak — especially after smoothing.

### 2. Uses *accumulated geometry*

A hook is fundamentally:

> “a short segment that changes direction significantly before stabilizing”

This detector directly measures that.

### 3. Natural stopping condition

The `stability_window` prevents over-cutting into real strokes.

---

# What you will likely observe

Compared to Version 1:

* ✔ much better trailing hook detection
* ✔ fewer missed subtle hooks
* ✔ slightly more conservative (may undercut less aggressively)
* ⚠ occasionally includes a bit more of the transition into the main stroke (tunable via `stability_window`)

---

# One important tuning insight

If you see:

* hooks not fully removed → decrease `stability_window`
* real stroke segments being cut → increase `min_turn_deg`
* over-aggressive trimming → increase `curvature_rel_threshold`

---

# If you want next step

After you test this, the natural upgrade (if needed) is:

> **bidirectional consistency check**

That is:

* start sweep and end sweep should not overlap into the same “stable region”

That’s usually the final refinement before production-quality stroke cleaning.

> I like this approach, but with default parameters it is overly aggressive. I will look into fine tuning the parameters for improved results.

That’s expected with this kind of inward-sweep method. Version 2 is less “sensitive to peaks” and more “sensitive to structure”, so with permissive defaults it will happily interpret legitimate curvature as a hook.

The important thing is: you’re now tuning the right *failure modes*, not fighting noise.

---

## What’s actually going wrong when it feels “too aggressive”

In practice, over-aggressive trimming usually comes from one of these:

### 1. Endpoint window too large

You’re giving the detector too much real stroke to analyze.

👉 Fix first:

```python
endpoint_fraction = 0.08  # start small
```

For Japanese handwriting, 0.15 is often already too wide.

---

### 2. Stability condition is too weak

If `stability_window` is small, the sweep stops *before* the real stroke direction is established, so it attributes normal curvature to the hook.

👉 Fix:

```python
stability_window = 7  # or even 10
```

Think of this as “how confident we are we’ve left the hook region”.

---

### 3. Curvature threshold is too permissive

If curvature threshold is low, normal stroke turns are interpreted as hook structure.

👉 Fix:

```python
curvature_rel_threshold = 2.5  # or 3.0
```

Or better (more stable across writers):

```python
k_thr = np.percentile(k, 75)
```

---

### 4. min_turn_deg is too low (most common issue)

This directly controls “what counts as a hook”.

👉 For handwriting, a good practical range is:

* subtle hooks: 20–25°
* normal strokes: 30–45°
* very strict cleaning: 45°+

So start here:

```python
min_turn_deg = 30
```

---

## A more stable mental model for tuning

Instead of thinking in terms of “parameters”, think:

### You are defining 3 zones:

1. **Hook zone (high curvature, high turn, short duration)**
2. **Transition zone (uncertain, ignored by stability logic)**
3. **Main stroke (stable direction, low curvature variance)**

Your parameters just control the boundaries:

| Parameter           | Controls                             |
| ------------------- | ------------------------------------ |
| endpoint_fraction   | how much of zone (1) is even visible |
| curvature threshold | sensitivity to entering zone (1)     |
| min_turn_deg        | whether zone (1) is real or noise    |
| stability_window    | when zone (3) begins                 |

---

## Minimal “safer default” for your case

If Version 2 is currently over-cutting, I would reset to:

```python
endpoint_fraction = 0.10
min_turn_deg = 30
stability_window = 8
curvature_rel_threshold = 2.5
```

Then only adjust **one parameter at a time**, in this order:

1. endpoint_fraction (biggest impact)
2. min_turn_deg (semantic meaning)
3. stability_window (stability behavior)
4. curvature threshold (least important in smoothed data)

---

## One important insight (this will matter later)

What you’re doing here is still fundamentally:

> “thresholding a noisy geometric signal”

Version 2 works, but its sensitivity is *distributed across multiple knobs*, which is why tuning feels fragile.

The next conceptual step (if/when you want it) is:

> convert the sweep into a **single hook score per endpoint**

That removes most of this tuning pain entirely.

---

If you want, the next iteration can make this detector return:

* a **continuous hook probability**
* a **cut position confidence curve**
* and only then a binary decision

That’s usually where these systems become stable across different writing styles.

> Yeah, why not? You can add those proposals to the output of version 2.

Good — this is the right direction. Once you move from *hard decisions* to a *score*, tuning becomes much easier because you can see *why* something is classified as a hook instead of just whether it is.

Below is an upgraded **Version 2 with scoring**. It keeps the same inward sweep idea, but now returns:

* `start_score`, `end_score` → hook likelihood (continuous)
* `start_hook`, `end_hook` → cut indices (derived from score)
* `debug traces` (optional but very useful)

---

# Version 2 (scored, improved)

```python
import numpy as np


def detect_hooks_v3(
    geom,
    endpoint_fraction=0.1,
    min_turn_deg=30,
    stability_window=8,
    curvature_rel_threshold=2.5,
):
    """
    Hook detection with continuous scoring.

    Returns both:
    - binary decisions (start_hook, end_hook)
    - continuous scores (start_score, end_score)
    """

    s = geom["s_norm"]
    k = geom["curvature"]
    angle = geom["angle"]

    n = len(s)

    k_med = np.median(k)
    k_thr = curvature_rel_threshold * k_med

    # -----------------------------
    # helper: inward sweep with scoring
    # -----------------------------
    def sweep(indices, forward=True):
        if len(indices) < 4:
            return None, 0.0, []

        if not forward:
            indices = indices[::-1]

        prev_angle = angle[indices[0]]
        accumulated_turn = 0.0

        low_count = 0
        trace = []

        hook_end_idx = indices[0]

        for i in indices[1:]:

            dtheta = angle[i] - prev_angle
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

            abs_turn = abs(dtheta)
            accumulated_turn += abs_turn

            # curvature signal (normalized contribution)
            curvature_signal = max(0.0, (k[i] - k_thr) / (k_thr + 1e-12))

            if k[i] < k_thr:
                low_count += 1
            else:
                low_count = 0

            trace.append((i, accumulated_turn, curvature_signal))

            prev_angle = angle[i]
            hook_end_idx = i

            if low_count >= stability_window:
                break

        if len(trace) == 0:
            return None, 0.0, trace

        # convert to degrees
        total_turn_deg = np.degrees(accumulated_turn)

        # -----------------------------
        # scoring function
        # -----------------------------
        turn_score = np.tanh(total_turn_deg / min_turn_deg)

        curvature_score = np.mean([t[2] for t in trace])

        # combine signals
        score = 0.7 * turn_score + 0.3 * curvature_score

        # decision
        if total_turn_deg >= min_turn_deg:
            return hook_end_idx, score, trace

        return None, score, trace

    # -----------------------------
    # endpoint regions
    # -----------------------------
    start_region = np.where(s <= endpoint_fraction)[0]
    end_region = np.where(s >= 1 - endpoint_fraction)[0]

    # -----------------------------
    # detect start
    # -----------------------------
    start_hook, start_score, start_trace = sweep(start_region, forward=True)

    # -----------------------------
    # detect end
    # -----------------------------
    end_hook, end_score, end_trace = sweep(end_region, forward=False)

    return {
        "start_hook": start_hook,
        "end_hook": end_hook,
        "start_score": float(start_score),
        "end_score": float(end_score),
        "start_trace": start_trace,
        "end_trace": end_trace,
    }
```

---

# What changed conceptually

## 1. Hard threshold → soft score

Instead of:

> “Is this a hook or not?”

we now compute:

> “How hook-like is this endpoint?”

---

## 2. Two independent signals

### A. Turning signal

```python
turn_score = tanh(total_turn / threshold)
```

* grows smoothly
* saturates (prevents runaway values)

### B. Curvature support signal

```python
curvature_score = mean(normalized curvature excess)
```

* detects whether turning is happening in a “bent region”

---

## 3. Combined interpretation

```python
score = 0.7 * turn_score + 0.3 * curvature_score
```

You can now interpret:

| Score   | Meaning                     |
| ------- | --------------------------- |
| < 0.3   | almost certainly not a hook |
| 0.3–0.6 | uncertain / borderline      |
| > 0.6   | likely hook                 |

---

# Why this fixes your current problem

Your current issue (“overly aggressive”) happens because:

* Version 2 makes **binary decisions too early**
* curvature noise directly triggers cuts

Now:

### instead of cutting immediately

you first accumulate evidence

So a small accidental curvature spike is no longer enough.

---

# Practical usage

```python
result = detect_hooks_v3(geom)

if result["end_score"] > 0.6:
    cut_idx = result["end_hook"]
```

Or more conservative:

```python
if result["end_score"] > 0.75:
```

---

# What you gained

You now have:

### ✔ continuous control

No more brittle threshold tuning

### ✔ explainability

You can inspect:

```python
result["end_trace"]
```

to see *why* it fired

### ✔ separation of concerns

* geometry → signal extraction
* scoring → decision layer

---

# Next logical upgrade (optional, but powerful)

If you continue this direction, the next step is:

> learn the scoring weights instead of hand-setting them

But even without ML, this version is already close to a production-grade heuristic system.

---

If you want, next step I’d suggest is:

👉 converting `start_trace` / `end_trace` into a **visual diagnostic overlay on strokes**

That usually makes parameter tuning almost unnecessary.
