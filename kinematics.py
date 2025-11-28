import numpy as np


def calculateMatrix(theta, d, a, alpha):
    """
    Compute a single Denavit-Hartenberg transformation matrix.
    
    Args:
        theta: rotation angle around z-axis (rad)
        d: translation along z-axis
        a: translation along x-axis
        alpha: rotation around x-axis (rad)
    
    Returns:
        4x4 homogeneous transformation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ], dtype=float)


def forwards_kinematics(theta_1, theta_2, theta_3, theta_4):
    """
    Compute the forward kinematics for a 4-DOF manipulator.
    
    Args:
        theta_1, theta_2, theta_3, theta_4: joint angles (rad)
    
    Returns:
        T_03: transformation up to joint 3
        T_04: transformation up to joint 4 (end-effector)
        T_05: transformation including tool offset
    """
    # DH transformations between consecutive joints
    T_01 = calculateMatrix(theta_1, 50, 0, np.pi/2)
    T_12 = calculateMatrix(theta_2, 0, 93, 0)
    T_23 = calculateMatrix(theta_3, 0, 93, 0)
    T_34 = calculateMatrix(theta_4, 0, 75, 0)  # real EE offset

    # Transformations up to joint 3 and 4
    T_03 = T_01 @ T_12 @ T_23
    T_04 = T_03 @ T_34

    # Wrist rotation (relative to joint 3)
    R_35 = np.array([
        [np.cos(theta_4), -np.sin(theta_4), 0, 0],
        [np.sin(theta_4),  np.cos(theta_4), 0, 0],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ], dtype=float)

    # Tool offset (relative to joint 3)
    T_35 = np.array([
        [1, 0, 0, 46],
        [0, 1, 0, 42],
        [0, 0, 1, 17],
        [0, 0, 0, 1]
    ], dtype=float)

    # End-effector transformation including tool
    T_05 = T_03 @ R_35 @ T_35
    return T_03, T_04, T_05


def inverseKinematics(o):
    """
    Compute IK for a 4-DOF manipulator (returns elbow-down configuration).
    
    Args:
        o : array-like, desired end-effector position [x, y, z]

    Returns:
        q_down : 4-element array of joint angles (rad) for elbow-down configuration
    """
    # Robot geometric parameters
    d4 = 76
    d1 = 50
    a2 = 93
    a3 = 93

    # Desired end-effector x-axis direction (fixed downward)
    x = [0, 0, -1]

    # Wrist center position
    x_c = o[0] - x[0] * d4
    y_c = o[1] - x[1] * d4
    z_c = o[2] - x[2] * d4

    # Base joint angle
    q0 = np.arctan2(y_c, x_c)

    # Distances in the vertical plane
    r_sq = x_c**2 + y_c**2
    s = z_c - d1

    # Compute q2 using law of cosines
    c2 = (r_sq + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    c2 = np.clip(c2, -1, 1)  # numerical stability

    s2_up = np.sqrt(1 - c2**2)
    s2_down = -s2_up

    # Elbow-up and elbow-down q2
    q2_up = np.arctan2(s2_up, c2)
    q2_down = np.arctan2(s2_down, c2)

    # q1 for both elbow configurations
    q1_up = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3 * s2_up, a2 + a3 * c2)
    q1_down = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3 * s2_down, a2 + a3 * c2)

    # q3 (wrist) for both configurations
    q3_up = np.arctan2(x[2], np.sqrt(x[0]**2 + x[1]**2)) - q1_up - q2_up
    q3_down = np.arctan2(x[2], np.sqrt(x[0]**2 + x[1]**2)) - q1_down - q2_down

    # Only return elbow-down
    q_down = np.array([q0, q1_down, q2_down, q3_down])

    return q_down
