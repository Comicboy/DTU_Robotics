import numpy as np

def calculateMatrix(theta, d, a, alpha):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ], dtype=float)

def forwards_kinematics(theta_1, theta_2, theta_3, theta_4):
    T_01 = calculateMatrix(theta_1, 50, 0, np.pi/2)
    T_12 = calculateMatrix(theta_2, 0, 93, 0)
    T_23 = calculateMatrix(theta_3, 0, 93, 0)
    T_34 = calculateMatrix(theta_4, 0, 50, 0)

    T_03 = T_01 @ T_12 @ T_23
    T_04 = T_03 @ T_34

    T_35 = np.array([
        [np.cos(theta_4), -np.sin(theta_4), 0, 35],
        [np.sin(theta_4),  np.cos(theta_4), 0, 45],
        [0,                0,               1, 0],
        [0,                0,               0, 1]
    ], dtype=float)

    T_05 = T_03 @ T_35
    return T_03, T_04, T_05

def inverseKinematics(x_dir, o, return_both=False):
    """
    x_dir: vettore direzione dell'end-effector (verso il basso = [0,0,-1])
    o: posizione obiettivo [x,y,z]
    """
    d4 = 50
    d1 = 50
    a2 = 93
    a3 = 93

    x_c = o[0] - x_dir[0]*d4
    y_c = o[1] - x_dir[1]*d4
    z_c = o[2] - x_dir[2]*d4

    q0 = np.arctan2(y_c, x_c)

    r_sq = x_c**2 + y_c**2
    s = z_c - d1

    c2 = (r_sq + s*s - a2*a2 - a3*a3) / (2*a2*a3)
    c2 = np.clip(c2, -1, 1)

    s2_up = np.sqrt(1 - c2*c2)
    s2_down = -s2_up

    q2_up = np.arctan2(s2_up, c2)
    q2_down = np.arctan2(s2_down, c2)

    q1_up = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*s2_up, a2 + a3*c2)
    q1_down = np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*s2_down, a2 + a3*c2)

    # Polso orientato secondo x_dir
    q3_up   = np.arctan2(-x_dir[1], -x_dir[2]) - q1_up - q2_up
    q3_down = np.arctan2(-x_dir[1], -x_dir[2]) - q1_down - q2_down

    if return_both:
        # q_down = gomito giù, q_up = gomito su
        return np.array([q0, q1_down, q2_down, q3_down]), np.array([q0, q1_up, q2_up, q3_up])

    return np.array([q0, q1_up, q2_up, q3_up])


def inverseKinematics_position(pos, x_dir=np.array([0,0,0]), return_both=False):
    q_down, q_up = inverseKinematics(x_dir, pos, return_both=True)
    
    if return_both:
        return q_up, q_down   # ritorna sempre [Elbow Up, Elbow Down]
    return q_up
