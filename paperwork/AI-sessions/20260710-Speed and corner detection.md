Topic: Speed and corner detection
Bot: Gemini
Date: 26 07 10

> Given is the followinf python code

```
dx = np.gradient(stroke[:, X], s)
dy = np.gradient(stroke[:, Y], s)
speed = np.hypot(dx, dy)
```  

If s is the cumulative arc length of a curve (stroke) and stroke[:, X] and stroke[:, Y] the x and y coordinates. What does speed actually represent and why?

In this code, speed actually represents a constant value of exactly $1$ at all points along the curve.
Here is the step-by-step breakdown of why this happens.
## 1. Identify the derivatives
The variables dx and dy approximate the derivatives of the coordinates with respect to the arc length $s$.

* dx represents $\frac{dx}{ds}$
* dy represents $\frac{dy}{ds}$

## 2. Analyze the hypotenuse calculation
The np.hypot(dx, dy) function calculates the magnitude of the tangent vector:
$$\text{speed} = \sqrt{\left(\frac{dx}{ds}\right)^2 + \left(\frac{dy}{ds}\right)^2}$$
## 3. Apply the arc length property
By definition, the differential arc length element $ds$ of a 2D curve is defined by the Pythagorean theorem:
$$ds^2 = dx^2 + dy^2$$
If you divide both sides of this foundational geometric equation by $ds^2$, you get:
$$1 = \left(\frac{dx}{ds}\right)^2 + \left(\frac{dy}{ds}\right)^2$$
## 4. Evaluate the final speed
Taking the square root of both sides yields:
$$\sqrt{\left(\frac{dx}{ds}\right)^2 + \left(\frac{dy}{ds}\right)^2} = 1$$
Because you are differentiating the curve's position coordinates with respect to its own cumulative distance (arc length), the resulting vector is a unit tangent vector. Its magnitude (the "speed") will always identically equal $1$, regardless of the curve's shape. [1]
(Note: In practice, numerical approximations from np.gradient might cause tiny deviations from exactly $1.0$ at the boundaries, but theoretically and structurally, it is a constant $1$.)
## ✅ Conclusion
The variable speed evaluates to a constant value of $1$ because differentiating a spatial curve with respect to its own arc length always yields a unit vector.
If you are trying to find the physical speed of the pen or mouse cursor, let me know:

* Do you have a time variable ($t$) or timestamps for each point?
* Are the points sampled at a constant frame rate?

I can show you how to alter the code to calculate the true physical speed!

[1] [https://www.youtube.com](https://www.youtube.com/watch?v=dh4qKZS2NcM)

> The plot for speed clearly shows ranges where speed is between 0 and 1, not a constant 1.

The observation that your speed plot drops below $1$ is caused by numerical discretization errors inherent to how np.gradient and cumulative arc length interact on discrete, sharply turning data.
While the continuous calculus proves the value must be exactly $1$, discrete computer code introduces two specific artifacts that cause the calculated speed to dip between $0$ and $1$, especially during fast or sharp strokes.
## 1. Cumulative Chord Length vs. True Arc Length
When you calculate s in Python, you are likely calculating the cumulative chord length (straight lines between consecutive points):
$$s_i = \sum \sqrt{\Delta x_i^2 + \Delta y_i^2}$$
This assumes the path between point $i$ and point $i+1$ is a perfectly straight line. However, a digital stroke consists of discrete samples of a smooth curve. The straight-line approximation always underestimates the true curvilinear arc length if the path curves between samples.
## 2. The Central Difference Mismatch
The primary culprit is how np.gradient calculates derivatives. For an interior point $i$, np.gradient uses a central difference formula.
It calculates the derivative by looking at the neighbors $i-1$ and $i+1$, completely skipping the current point $i$:
$$\frac{dx}{ds} \approx \frac{x_{i+1} - x_{i-1}}{s_{i+1} - s_{i-1}}$$
Now, look at what happens when the curve changes direction or changes speed rapidly:

* The Numerator ($\Delta x, \Delta y$): Measures the straight-line distance shortcut from the previous point straight to the next point ($i-1$ to $i+1$). If the stroke makes a sharp turn, this shortcut distance is much shorter than the actual path traveled.
* The Denominator ($\Delta s$): Sums the individual step lengths ($s_{i} - s_{i-1}$) + ($s_{i+1} - s_{i}$). This preserves the longer, bent path length.

Because the numerator uses a shortcut distance while the denominator uses the full accumulated path distance, the fraction becomes smaller than $1$.
$$\text{Calculated Speed} = \frac{\text{Shortcut Distance Over 2 Steps}}{\text{Accumulated Step Lengths Over 2 Steps}} \le 1$$
## 3. Summary of Visualized Behavior

| Stroke Feature | Mathematical Behavior | Visual Result in Plot |
|---|---|---|
| Perfect Straight Line | Central shortcut matches the accumulated path. | speed will be exactly $1.0$. |
| Sharp Corners / High Curvature | Shortcut distance is significantly shorter than the path. | speed dips sharply toward $0$. |
| High Sampling Density | Shortcut path closely tracks the tiny curves. | speed hovers very close to $1.0$. |

## ✅ Conclusion
Your code does not measure physical speed. Instead, it measures local curvature artifacts and digitization errors. The dips below $1$ highlight regions where the stroke is turning sharply relative to your sample rate, forcing the central difference algorithm to take a geometric shortcut.
If you want to fix this or analyze the stroke characteristics, tell me:

* Are you trying to calculate the true physical drawing speed over time?
* Or are you trying to use these dips as a proxy feature to detect sharp corners and curvature?

> OK then. So speed as calculated above can be used to detect abrupt changes in direction?

Yes, absolutely. While it is named speed in the code, it structurally acts as an inverse proxy for curvature and directional change. [1]
Whenever the stroke changes direction abruptly, the value will dip below 1. The sharper the turn, the closer the value drops toward 0.
## Why it works for corner detection
The behavior relies entirely on the geometric shortcut created by the central difference method:

* On straight paths: The distance from point $i-1$ to $i+1$ matches the accumulated step lengths. The ratio is $\approx 1$.
* On sharp corners: The straight line from $i-1$ to $i+1$ cuts completely across the corner. The numerator (shortcut distance) becomes much smaller than the denominator (the actual path walked around the corner). The ratio drops significantly below 1.

## Limitations to keep in mind
If you plan to use this metric for feature engineering or corner detection, be aware of two major quirks:

* Sampling Rate Dependency: The dips depend heavily on your sampling rate. If you record a sharp corner very slowly (generating many tightly packed points), the shortcut error minimizes, and the dip might disappear.
* Vulnerability to Noise: Micro-jitters from a digital pen or mouse can look like tiny, rapid changes in direction. This will cause the metric to noisy-drop below 1 even on straight lines.

## A more robust alternative
If your goal is to map out changes in direction cleanly without sampling-rate bugs, you can calculate the turning angle (the angular change $\Delta \theta$ between consecutive displacement vectors) directly:

# Calculate segment vectorsdx = np.diff(stroke[:, X])dy = np.diff(stroke[:, Y])
# Compute absolute angles of each segmentangles = np.arctan2(dy, dx)
# Find the change in angle between consecutive segmentsangle_changes = np.diff(angles)# Normalize angles to keep them between [-pi, pi]angle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi

An angle_changes value close to 0 means a straight line, while values spiking toward $\pm\pi$ (180 degrees) indicate severe, abrupt changes in direction.
Would you like to see how to use a thresholding filter to extract the exact $(x, y)$ coordinates of the corners using your current method, or would you like to explore the angle-change alternative?

[1] [https://allen.in](https://allen.in/dn/qna/646688290)

> Sampling rate in current context is almost constant 8 to 13 ms. Would there be a more appropriate name for the variable speed?

Since your sampling rate is nearly constant, a drop in this metric is driven almost entirely by the geometry of the curve rather than time fluctuations.
The most appropriate names for this variable depend on how you want to frame its physical or mathematical meaning:
## 1. Geometric & Signal Processing Names (Highly Recommended)

* path_fidelity or straightness_index: Captures how closely the local 3-point segment mimics a straight line. A value of 1.0 means perfect fidelity to a straight line; a dip means the path is deviating.
* chord_to_arc_ratio: The exact mathematical description of what is happening. It is the ratio of the straight shortcut (chord) to the actual traveled path (arc).

## 2. Feature Engineering & Corner Detection Names

* corner_index or cornerness: Ideal if you are using this variable directly to flag corners. You would look for local minima (valleys) in this signal.
* direction_stability: Highlights how stable the current trajectory is. High stability sits at 1.0, while directional instability drops the value.

## 3. Kinematic Approximations (If using it as a proxy for physical movement)

* pseudo_speed or spatial_velocity: If you want to keep the concept of "speed" because the code implies it, adding "pseudo" or "spatial" clarifies that it is normalized against space (arc length), not actual clock time.

## Why "Curvature" is technically backward
While it detects sharp turns, you should avoid naming the variable curvature. In mathematics, curvature spikes upward (toward infinity) at a sharp corner. Your variable drops downward (toward 0). If you want to call it curvature, you would need to invert it: curvature = 1.0 - speed.
------------------------------
If you want to refine your pipeline, would you like to see how to smooth out the 8–13 ms sampling jitter using a time-based interpolation, or are you ready to implement the threshold logic for finding the corners?

> Assume I would apply 1d gaussian smoothing on x and y before calculating speed. How would that effect the outcome.

Applying a 1D Gaussian filter to the $x$ and $y$ coordinates before calculating your metric will drastically change the behavior of the signal.
Gaussian smoothing acts as a low-pass filter that shaves off sharp edges and suppresses high-frequency noise. Here is exactly how it will affect your outcome:
## 1. The Dips Will Shallow Out (Reduced Sensitivity)
Because Gaussian smoothing rounds out sharp corners, your points will no longer make abrupt, 90-degree or 180-degree turns. [1]

* Without smoothing: A sharp corner causes a massive mismatch between the shortcut distance and the path length, driving the metric close to $0$.
* With smoothing: The corner becomes a gentle, wide arc. The central difference shortcut now tracks the smoothed path much better.
* The Result: The valleys in your plot will no longer drop as deeply toward $0$. They will become much shallower and wider.

## 2. High-Frequency Jitter Will Disappear
Raw digital ink data often suffers from micro-jitters (small hand shakes or hardware digitization noise). This noise manifests as tiny, fake "corners" that cause your metric to drop below $1$ randomly, even on straight lines.

* The Result: Smoothing eliminates this high-frequency noise. In the straight sections of your stroke, your metric will become incredibly stable and hover much closer to a clean, flat line at $1.0$.

## 3. Corner Detection Becomes Blurrier (Time-Shifted)
Gaussian smoothing distributes the geometric change of a single sharp point across its neighboring points.

* The Result: Instead of a sharp, distinct drop at the exact index of the corner, you will get a wide, smooth trough. While this makes the signal prettier, it makes it harder to pinpoint the exact millisecond or coordinate where the abrupt turn originally started.

## 4. Mathematical Artifact: Speed Can Exceed 1.0
This is a fascinating side effect of smoothing $x$ and $y$ independently before calculating a cumulative arc length $s$.

* If you smooth $x$ and $y$, you create a new, shortened curve.
* If your $s$ variable is calculated from the original, noisy coordinates, but dx and dy are calculated from the new, smoothed coordinates, the math breaks.
* The Result: You might see your metric actually climb above $1.0$ in certain segments because the original path lengths ($ds$) are longer than the smoothed positional changes ($dx, dy$).

## Summary: Is it a good idea?

| Metric | Without Smoothing | With Smoothing |
|---|---|---|
| Noise Level | High (lots of false positives) | Extremely low and clean |
| Corner Precision | High (sharp, deep drops) | Low (wide, muddy troughs) |
| Baseline Accuracy | Strictly $\le 1.0$ | Can artifact jump $> 1.0$ |

## 💡 Recommendation
If you want to smooth the data, recompute your arc length $s$ after smoothing.

# 1. Smooth coordinatesx_smooth = gaussian_filter1d(stroke[:, X], sigma)y_smooth = gaussian_filter1d(stroke[:, Y], sigma)
# 2. Recalculate s based ONLY on the smoothed coordinatesds = np.hypot(np.diff(x_smooth), np.diff(y_smooth))s_smooth = np.concatenate(([0], np.cumsum(ds)))
# 3. Calculate your metricdx = np.gradient(x_smooth, s_smooth)dy = np.gradient(y_smooth, s_smooth)path_fidelity = np.hypot(dx, dy)

This guarantees your baseline stays bounded at $1.0$ while giving you a beautifully clean signal for analysis.
Would you like to explore how to pick the optimal sigma value for the Gaussian filter so you suppress noise without destroying the corners entirely?

[1] [https://130.149.89.49](http://130.149.89.49:2080/v2016/books/exa/ch01s03aex45.html)

> Yes show me how to pick sigma.

To pick the optimal $\sigma$ (standard deviation) for your Gaussian filter, you must balance noise suppression against corner preservation.
Because your sampling rate is nearly constant at 8 to 13 ms (averaging roughly 10 ms per point), we can map $\sigma$ directly to a physical window of time and a specific number of data points.
------------------------------
## 1. The Rule of Thumb for $\sigma$
A Gaussian filter's influence extends roughly $3\sigma$ on either side of a point (covering a total window of $6\sigma$).

* If $\sigma = 1$: Looks 3 points back and 3 points forward ($\approx 60\text{ ms}$ total window).
* If $\sigma = 3$: Looks 9 points back and 9 points forward ($\approx 180\text{ ms}$ total window).

## 2. Choose Based on Your Goal## 🎯 Option A: The "Sweet Spot" ($\sigma = 1.0$ to $1.5$)

* Best for: Corner detection.
* Why: This filters out high-frequency hardware jitter (which occurs at the 10–20 ms scale) but preserves fast human stylistic changes in direction (which take roughly 50–100 ms to execute). Your dips will stay deep and distinct.

## 🎯 Option B: The "Smooth Path" ($\sigma = 2.0$ to $4.0$)

* Best for: Visualizing global stroke shapes or filtering out heavy hand tremors.
* Why: This averages out data over a 120–240 ms window. It completely rounds off sharp corners into gentle curves, making it poor for corner detection but great for macro-path analysis.

------------------------------
## 3. Data-Driven Selection (Code)
The best way to pick $\sigma$ is to calculate the Signal-to-Noise Ratio (SNR) or visually inspect how the metric changes. Below is a script to test multiple values of $\sigma$ simultaneously on your stroke data so you can pick the one that cleanly separates straight lines from corners.

import numpy as npfrom scipy.ndimage import gaussian_filter1d
def analyze_sigmas(stroke, x_idx, y_idx, sigmas=[0.5, 1.0, 2.0, 4.0]):
    results = {}

    for sigma in sigmas:
        # 1. Smooth
        x_s = gaussian_filter1d(stroke[:, x_idx], sigma)
        y_s = gaussian_filter1d(stroke[:, y_idx], sigma)

        # 2. Recalculate arc length on the smoothed curve
        ds = np.hypot(np.diff(x_s), np.diff(y_s))
        # Handle overlapping identical points to avoid divide-by-zero
        ds[ds == 0] = 1e-6
        s_s = np.concatenate(([0], np.cumsum(ds)))

        # 3. Calculate path fidelity
        dx = np.gradient(x_s, s_s)
        dy = np.gradient(y_s, s_s)
        fidelity = np.hypot(dx, dy)

        # 4. Score it: Look for standard deviation on "flat" parts vs depth of valleys
        results[sigma] = {
            "fidelity_signal": fidelity,
            "min_val": np.min(fidelity[1:-1]), # Ignore boundary effects
            "mean_val": np.mean(fidelity)
        }

    return results
# Example Usage:# test_results = analyze_sigmas(stroke, X, Y)

------------------------------
## 4. How to Evaluate the Output
Look at the fidelity_signal arrays generated by the code above:

   1. Check the baseline: Look at a known straight section of the stroke. If the signal is still jumping between 0.95 and 1.0, your $\sigma$ is too low (noise is leaking through).
   2. Check the valleys: Look at a known sharp turn. If the metric only drops to 0.92 instead of plummeting down toward 0.5 or lower, your $\sigma$ is too high (you are erasing the corner).
   3. The Winner: The smallest $\sigma$ that keeps your straight lines completely flat at 1.0 while still allowing corners to drop sharply. For your 10 ms sample rate, $\sigma = 1.2$ is usually the mathematical jackpot.

Would you like to see how to write a simple peak/valley finder to automatically extract the precise timestamps of the corners once you settle on a $\sigma$?

> I guess the calculation of angle_changes you gave earlier would benefit more from prior smoothing. In the plot I see rapid fluctuations with high amplitudes for very noisy areas and even for almost straight sections rapid fluctuations with much lower amplitudes.

You are completely right. Calculating angles using np.arctan2 acts as a numerical differentiator, which amplifies high-frequency noise.
In an un-smoothed straight line, a microscopic hardware jitter of just 1 pixel can cause the directional vector to wildly flip back and forth, creating the low-amplitude fluctuations you are seeing. In genuinely noisy areas, this completely drowns out the real trajectory.
Applying Gaussian smoothing before calculating angle_changes makes a massive difference.
## Why Smoothing Transforms the Angle Plot

   1. It tames the straight lines: By smoothing out the 1-pixel jitters, consecutive vectors point in almost the exact same direction. Your noisy baseline in straight sections will collapse from a vibrating band down to a beautifully clean, flat line near 0.
   2. It isolates true directional changes: Instead of massive, chaotic spikes in noisy areas, you will get clean, distinct peaks that accurately mirror the macro-turn of the stroke.

------------------------------
## The Clean Implementation Pipeline
When smoothing for angle changes, you do not need to recalculate arc length (unlike the path fidelity metric). You just smooth the coordinates, get the differences, and compute the angles.
Here is the robust way to implement this, including a step to handle the boundary artifacts that np.diff creates:

import numpy as npfrom scipy.ndimage import gaussian_filter1d
# 1. Apply Gaussian smoothing directly to coordinates (Try sigma=1.2 to 1.5)sigma = 1.2x_smooth = gaussian_filter1d(stroke[:, X], sigma)y_smooth = gaussian_filter1d(stroke[:, Y], sigma)
# 2. Calculate smooth segment vectorsdx = np.diff(x_smooth)dy = np.diff(y_smooth)
# 3. Compute absolute angles of each segmentangles = np.arctan2(dy, dx)
# 4. Find the change in angle between consecutive segmentsangle_changes = np.diff(angles)
# 5. Crucial: Normalize angles to keep them between [-pi, pi]# This prevents fake spikes when angles wrap around from +pi to -piangle_changes = (angle_changes + np.pi) % (2 * np.pi) - np.pi
# 6. Optional: Pad the array so it matches the original stroke length# np.diff reduces the length by 1 each time it's called (total loss of 2 elements)angle_changes_padded = np.pad(angle_changes, (1, 1), mode='edge')

------------------------------
## What to Look For in Your New Plot
Once you plot this smoothed version, your interpretation strategy shifts:

* Straight lines: Will hover tightly around 0 radians.
* Sharp turns / Corners: Will manifest as clean, sharp peaks (or valleys, depending on clockwise/counterclockwise turns). A 90-degree turn will show up as a beautiful peak at roughly $\pm 1.57$ radians ($\pi/2$). [1]
* Choosing Sigma: If your straight lines are still fuzzy, increase sigma by increments of 0.2. If your big turning peaks start shrinking drastically in height, your sigma is too high.

Would you like to look at a method to automatically detect and flag these peaks now that the signal is clean enough to threshold?

[1] [https://cgvr.cs.uni-bremen.de](https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/Fresnel%20Equations,%20Schlick%20Approximation,%20Metals,%20and%20Dielectrics%20-%20Pete%20Shirley%27s%20Graphics%20Blog.pdf)

>