import numpy as np
import xml.etree.ElementTree as ET
from svg.path import Path, parse_path
from shapely.geometry import LineString

def clamp(n, n_min, n_max):
    return max(n_min, min(n, n_max))


def extract_paths(content: str) -> Path:
    """
    Extract and parse all paths in a given SVG string.
    """

    tag = "path"
    namespace = "{http://www.w3.org/2000/svg}"
    xpath = ".//" + namespace + tag
    root = ET.fromstring(content)
    return [parse_path(path.attrib['d']) for path in root.findall(xpath)]

def normalize_strokes(strokes):
    all_normalized = []

    for stroke in strokes:
        normalied = [[coord[0] / 110, coord[1] / 110] for coord in stroke]
        all_normalized.append(normalied)

    return all_normalized


def resample_paths_to_fixed(parsed_paths: [Path], max_points: int) -> np.ndarray:
    """
    Resamples a list of svg.path Path objects to strictly fit a global point budget.

    Parameters:
        parsed_paths (list): A list of Path objects resulting from svg.path.parse_path().
        max_points (int): Strict ceiling for the total number of points across the Kanji.

    Returns:
        list of np.ndarray: List of shape (N, 2) arrays containing the resampled [x, y] coordinates.
    """
    # 1. Filter out empty paths or standalone Move/Close segments with zero length
    valid_paths = []
    stroke_lengths = []

    for path in parsed_paths:
        # Ignore empty path definitions
        try:
            # svg.path calculates lengths via geometric approximation for curves
            length = float(path.length())
        except Exception:
            length = 0.0

        valid_paths.append(path)
        stroke_lengths.append(max(length, 1e-5)) # Avoid divide-by-zero for zero-length artifacts

    num_strokes = len(valid_paths)
    if num_strokes == 0:
        return []

    # 2. Enforce structural minimum limits (Each stroke requires start/end point)
    if max_points < 2 * num_strokes:
        raise ValueError(f"Point budget ({max_points}) is too low to represent {num_strokes} strokes. Minimum required: {2 * num_strokes}.")

    total_length = sum(stroke_lengths)

    # 3. Allocate point budgets proportionally based on stroke length
    base_points_used = 2 * num_strokes
    pool_points = max_points - base_points_used

    allocations = []
    for length in stroke_lengths:
        proportion = length / total_length
        allocated = int(np.floor(proportion * pool_points))
        allocations.append(2 + allocated) # Base points + proportional pool share

    # 4. Handle rounding remainders systematically (distribute to longest strokes first)
    leftover_points = max_points - sum(allocations)
    if leftover_points > 0:
        longest_indices = np.argsort(stroke_lengths)[::-1]
        for i in range(leftover_points):
            allocations[longest_indices[i % num_strokes]] += 1

    # 5. Extract points using parametric step functions
    resampled_kanji = []
    for path, p_target in zip(valid_paths, allocations):
        # Generate linearly spaced parametric evaluation points from 0.0 to 1.0
        t_values = np.linspace(0.0, 1.0, p_target)

        stroke_points = []
        for t in t_values:
            # path.point(t) returns a complex number: real=X, imag=Y
            complex_pt = path.point(t)
            stroke_points.append([complex_pt.real / 110, complex_pt.imag / 110])

        features = np.hstack((stroke_points, np.ones((len(stroke_points), 1))))
        features[-1, 2] = 0  # Pen-Up Signal
        resampled_kanji.append(features)

    return np.vstack(resampled_kanji)


def resample_path_to_max(path: Path, n_points: int = 16):
    """
    Resample a SVG path to a maximum number of points.
    """

    normalized_point = lambda t: [path.point(t).real / 110, path.point(t).imag / 110]

    # Approximate arc length
    points = np.array([
        normalized_point(t)
        for t in np.linspace(0.0, 1.0, 32)
    ])

    seg_diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]
    n_points = clamp(round(n_points * total_length), 2, n_points)

    return np.array([
        normalized_point(t)
        for t in np.linspace(0.0, 1.0, n_points)
    ])

"""
ChatGPT prompt:
Implement a function in Python which resamples a list of 2D line strings
into a list of NumPy arrays. The resulting sum of all points of the
result must be exactly n_point which is given as a function argument.

ChatGPT answer:
Here is a robust Python function that handles this task.
It uses NumPy to calculate geographic line lengths, fairly distributes
the exact budget of target vertices using the Largest Remainder
Method (Hamilton Method), and uses linear interpolation (np.interp)
to generate perfectly spaced coordinates.
"""

def resample_strokes_to_fixed(lines: list, n_points: int) -> list:
    """
    Resamples a list of 2D line strings into a list of NumPy arrays such that
    the total sum of all points across all lines exactly equals n_points.

    Parameters:
        lines (list): List of line strings (each list-like or array of shape (M, 2)).
        n_points (int): The exact cumulative number of vertices required in the output.

    Returns:
        list: A list of NumPy arrays of shape (K, 2) where sum(K) == n_points.
    """

    lines = normalize_strokes(lines)

    n_lines = len(lines)
    if n_lines == 0:
        return []
    if n_points < 2 * n_lines:
        raise ValueError(f"n_points ({n_points}) must be at least {2 * n_lines} "
                         f"to guarantee a start and end point for each line.")

    # Step 1: Calculate cumulative and total lengths for all line strings
    line_data = []
    total_network_length = 0.0

    for line in lines:
        arr = np.asarray(line, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Each line string must be an array-like structure of 2D points (M, 2).")

        # Distance between consecutive vertices
        segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        line_length = cumulative_distances[-1]

        line_data.append((arr, cumulative_distances, line_length))
        total_network_length += line_length

    # Step 2: Allocate points safely using the Largest Remainder Method
    # Initialize every line with at least 2 points (start and end)
    allocated_points = np.full(n_lines, 2, dtype=int)
    remaining_points = n_points - (2 * n_lines)

    if remaining_points > 0 and total_network_length > 0:
        # Calculate proportional shares for the remaining points
        ideal_shares = (np.array([data[2] for data in line_data]) / total_network_length) * remaining_points
        floor_shares = np.floor(ideal_shares).astype(int)

        allocated_points += floor_shares
        leftover_fractional_points = remaining_points - np.sum(floor_shares)

        # Distribute leftover points to lines with the highest fractional remainders
        if leftover_fractional_points > 0:
            remainders = ideal_shares - floor_shares
            priority_indices = np.argsort(remainders)[::-1]
            for i in range(leftover_fractional_points):
                allocated_points[priority_indices[i]] += 1

    # Step 3: Linearly interpolate coordinates based on the allocated budget
    resampled_lines = []
    for (coords, cum_dist, total_len), num_pts in zip(line_data, allocated_points):
        if total_len == 0:
            # Handle degenerate lines (all vertices overlap at one single spot)
            resampled = np.repeat(coords[0:1], num_pts, axis=0)
        else:
            # Generate equally spaced steps along the absolute line length
            target_distances = np.linspace(0.0, total_len, num_pts)

            # Linearly map target distances back to X and Y coordinates
            x_new = np.interp(target_distances, cum_dist, coords[:, 0])
            y_new = np.interp(target_distances, cum_dist, coords[:, 1])
            resampled = np.column_stack((x_new, y_new))

        features = np.hstack((resampled, np.ones((len(resampled), 1))))
        features[-1, 2] = 0  # Pen-Up Signal
        resampled_lines.append(features)

    return np.vstack(resampled_lines)


def parse_entry_to_fixed(archive, entry, n_points: int = 16):
    content = archive.read(entry).decode("utf-8")
    paths = extract_paths(content)
    return resample_paths_to_fixed(paths, n_points)


def parse_entry_to_max(archive, entry, n_points: int = 16):
    content = archive.read(entry).decode("utf-8")
    paths = extract_paths(content)

    strokes = []
    for path in paths:
        resampled = resample_path_to_max(path, n_points)
        features = np.hstack((resampled, np.ones((len(resampled), 1))))
        features[-1, 2] = 0  # Pen-Up Signal
        strokes.append(features)

    return np.vstack(strokes)
