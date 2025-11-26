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
    # Compute DH transformations between consecutive joints
    T_01 = calculateMatrix(theta_1, 50, 0, np.pi/2)
    T_12 = calculateMatrix(theta_2, 0, 93, 0)
    T_23 = calculateMatrix(theta_3, 0, 93, 0)
    T_34 = calculateMatrix(theta_4, 0, 50, 0)

    # Transformation up to joint 3 and 4
    T_03 = T_01 @ T_12 @ T_23
    T_04 = T_03 @ T_34

    # Tool offset transformation (end-effector relative to joint 3)
    T_35 = np.array([
        [np.cos(theta_4), -np.sin(theta_4), 0, 35],
        [np.sin(theta_4),  np.cos(theta_4), 0, 45],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ], dtype=float)

    # Complete end-effector transformation including tool
    T_05 = T_03 @ T_35
    return T_03, T_04, T_05


def inverseKinematics(x_dir, o, return_both=False):
    """
    Compute inverse kinematics for a 4-DOF manipulator.
    
    Args:
        x_dir: desired orientation vector for the wrist
        o: desired end-effector position [x, y, z]
        return_both: if True, returns both elbow-up and elbow-down solutions
    
    Returns:
        Array of joint angles [q0, q1, q2, q3] (rad)
    """
    # Robot geometric parameters
    d4 = 50
    d1 = 50
    a2 = 93
    a3 = 93

    # Compute wrist center position
    x_c = o[0] - x_dir[0]*d4
    y_c = o[1] - x_dir[1]*d4
    z_c = o[2] - x_dir[2]*d4

    # Base rotation
    q0 = np.arctan2(y_c, x_c)

    # Distances for planar IK
    r_sq = x_c**2 + y_c**2
    s = z_c - d1

    # Compute q2 using the law of cosines
    c2 = (r_sq + s*s - a2*a2 - a3*a3) / (2*a2*a3)
    c2 = np.clip(c2, -1, 1)  # numerical safety

    s2_up = np.sqrt(1 - c2*c2)
    s2_down = -s2_up

    q2_up = np.arctan2(s2_up, c2)
    q2_down = np.arctan2(s2_down, c2)

    # Compute q1 for both configurations
    q1_up = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*s2_up, a2 + a3*c2)
    q1_down = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*s2_down, a2 + a3*c2)

    # Compute wrist joint angle based on desired x_dir
    q3_up   = np.arctan2(x_dir[2], np.sqrt(x_dir[0]**2 + x_dir[1]**2)) - q1_up - q2_up
    q3_down = np.arctan2(x_dir[2], np.sqrt(x_dir[0]**2 + x_dir[1]**2)) - q1_down - q2_down

    if return_both:
        return np.array([q0, q1_down, q2_down, q3_down]), np.array([q0, q1_up, q2_up, q3_up])

    return np.array([q0, q1_up, q2_up, q3_up])


def inverseKinematics_position(pos, x_dir=np.array([0,0,-1]), return_both=False):
    """
    Wrapper for inverse kinematics using position only.
    
    Args:
        pos: desired end-effector position [x, y, z]
        x_dir: desired wrist orientation vector (default zero)
        return_both: if True, returns both elbow-up and elbow-down solutions
    
    Returns:
        q_up, q_down if return_both=True
        else q_up only
    """
    q_down, q_up = inverseKinematics(x_dir, pos, return_both=True)
    
    if return_both:
        return q_up, q_down   # always returns [Elbow Up, Elbow Down]
    return q_up


    


    ##### DO NOT DELETE THIS PART
    # Wrist oriented with x_dir
    #q3_up   = np.arctan2(-x_dir[1], -x_dir[2]) - q1_up - q2_up
    #q3_down = np.arctan2(-x_dir[1], -x_dir[2]) - q1_down - q2_down