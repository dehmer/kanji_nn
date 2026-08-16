import numpy as np

def detect_curvature_spikes(stroke, kappa_threshold=20.0, velocity_threshold=0.005):
    """
    Identifies indices in a stroke that exhibit 'staircase' patterns
    based on high curvature and low velocity.

    Parameters:
    - stroke: np.array of shape (N, 2) representing [x, y] coordinates.
    - kappa_threshold: Curvature value above which a spike is flagged.
    - velocity_threshold: Velocity below which a point is considered 'slow'.

    Returns:
    - spike_indices: List of indices where staircase artifacts are detected.
    """
    # Ensure stroke is a numpy array
    N = stroke.n_points
    if N < 3:
        return []

    print("n_points", N)

    # Calculate first derivatives (velocity)
    dx = np.diff(stroke.xy[:, 0])
    dy = np.diff(stroke.xy[:, 1])
    v_x = dx
    v_y = dy

    # Calculate magnitude of velocity
    velocity = np.sqrt(v_x**2 + v_y**2)
    print("velocity", len(velocity))

    # Calculate second derivatives (acceleration)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    a_x = ddx
    a_y = ddy

    # Curvature calculation: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    # We use the indices that align between first and second derivatives
    num_points = len(v_x) - 1
    kappa = np.zeros(num_points)

    for i in range(num_points):
        denominator = (v_x[i]**2 + v_y[i]**2)**(1.5)
        if denominator > 0:
            kappa[i] = abs(v_x[i] * a_y[i] - v_y[i] * a_x[i]) / denominator
        else:
            kappa[i] = 0

    print("kappa", len(kappa))

    # Identify spikes: High curvature AND Low velocity
    spike_indices = []
    for i in range(len(kappa)):
        if kappa[i] > kappa_threshold and velocity[i] < velocity_threshold:
            spike_indices.append(i + 1) # Offset to match original stroke index

    curvature_spikes = spike_indices
    print("curvature_spikes", curvature_spikes)

    return stroke.clone(
        props={"spike_indices": spike_indices},
    )
