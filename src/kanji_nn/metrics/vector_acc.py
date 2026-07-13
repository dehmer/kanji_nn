import numpy as np

def vector_acc(stroke):
    """
    Calculates the magnitude of the true acceleration vector.
    Accounts for both changes in speed and changes in direction.
    """
    # 1. First derivatives (Velocities)
    vx = np.gradient(stroke.x, stroke.t)
    vy = np.gradient(stroke.y, stroke.t)

    # 2. Second derivatives (Accelerations)
    ax = np.gradient(vx, stroke.t)
    ay = np.gradient(vy, stroke.t)
    axy = np.column_stack((ax, ay))

    # 3. Combine using hypotenuse for total acceleration magnitude
    am = np.hypot(ax, ay)
    return stroke.clone(features={"vx": vx, "vy": vy, "axy": axy, "ax": ax, "ay": ay, "am": am})
