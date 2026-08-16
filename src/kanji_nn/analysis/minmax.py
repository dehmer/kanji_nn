def minmax():
    min_duration, max_duration = float('inf'), float('-inf')
    min_s, max_s = float('inf'), float('-inf')
    min_points, max_points = float('inf'), float('-inf')

    def inner(stroke):
        nonlocal min_duration
        nonlocal max_duration
        nonlocal min_s
        nonlocal max_s
        nonlocal min_points
        nonlocal max_points

        min_duration = min(min_duration, stroke.t[-1] - stroke.t[0])
        max_duration = max(max_duration, stroke.t[-1] - stroke.t[0])
        min_s = min(min_s, stroke.features["raw:s"][-1])
        max_s = max(max_s, stroke.features["raw:s"][-1])
        min_points = min(min_points, stroke.n_points)
        max_points = max(max_points, stroke.n_points)
        print(f"{stroke.literal}/{stroke.stroke_index}", min_duration, max_duration, min_s, max_s, min_points, max_points)
        return stroke

    return inner
