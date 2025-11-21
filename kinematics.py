import numpy as np
from control import MOTOR_LIMITS

def calculateMatrix(theta,d,a,alpha):
    A = np.array([[np.cos(theta) , -np.sin(theta)*np.cos(alpha) , np.sin(theta)*np.sin(alpha) , a*np.cos(theta)],
                  [np.sin(theta), np.cos(theta)*np.cos(alpha),-np.cos(theta)*np.sin(alpha),a*np.sin(theta)],
                  [0,np.sin(alpha),np.cos(alpha),d],
                  [0,0,0,1]], dtype=float)
    return A#np.array([[cosd(theta) , -sind(theta)*cosd(alpha) , sind(theta)*sind(alpha) , a*cosd(theta)],[sind(theta) cosd(theta)*cosd(alpha) -cosd(theta)*sind(alpha) a*sind(theta)],[0 sind(alpha) cosd(alpha) d],[0 0 0 1]])

def forwards_kinematics(theta_1,theta_2,theta_3,theta_4):
    T_01 = calculateMatrix(theta_1,50,0,np.pi/2)
    T_12 = calculateMatrix(theta_2,0,93,0)
    T_23 = calculateMatrix(theta_3,0,93,0)
    T_34 = calculateMatrix(theta_4,0,50,0)
    
    T_03=T_01@T_12@T_23
    T_04=T_01@T_12@T_23@T_34
    
    T_35=np.array([[np.cos(theta_4),-np.sin(theta_4),0,35],[np.sin(theta_4),np.cos(theta_4),0,45],[0,0,1,0], [0,0,0,1]],dtype=float)
    T_05 = T_01@T_12@T_23@T_35

    return T_03,T_04,T_05



import numpy as np

# Limiti dei servo in gradi
SERVO_LIMITS_DEG = {
    1: (60, 240),
    2: (51, 150),
    3: (40, 220),
    4: (135, 240)
}

def calculate_circle_step(phi):
    p_c = np.array([150, 0, 120])  # Center of the circle
    radius = 32  # Radius of the circle
    rot = np.array([0,np.cos(phi),np.sin(phi)])
    return p_c + radius * rot

def limit_angle(angle_deg, servo_id):
    low, high = SERVO_LIMITS_DEG[servo_id]
    return min(max(angle_deg, low), high)




def inverseKinematics(pos, x_dir_z):
    """
    Compute 4-DOF inverse kinematics for a 4R arm.
    
    Parameters
    ----------
    pos : array-like, shape (3,)
        Desired end-effector position [x, y, z].
    x_dir_z : float
        z-component of the end-effector's x-axis (orientation along stylus).
    
    Returns
    -------
    q_rel : ndarray, shape (4,)
        Joint angles [q1, q2, q3, q4] in radians relative to servo zero (0=center),
        preferred elbow-up configuration.
    """

    # Robot parameters
    a2, a3, a4, d1 = 93, 93, 50, 50
    o_04 = np.array(pos)
    x_04 = x_dir_z

    # Compute wrist offset
    d_4z = a4 * x_04
    u = np.sqrt(a4**2 - d_4z**2)

    # Base angle (theta1)
    theta1_deg = np.rad2deg(np.arctan2(o_04[1], o_04[0]))

    # Wrist projection
    x43 = u * np.cos(np.deg2rad(theta1_deg))
    y43 = u * np.sin(np.deg2rad(theta1_deg))

    # Wrist center
    xc = o_04[0] - x43
    yc = o_04[1] - y43
    zc = o_04[2] - d_4z

    # Planar distance
    r = np.sqrt(xc**2 + yc**2)
    s = zc - d1

    # Elbow angles
    c3 = (r**2 + s**2 - a2**2 - a3**2) / (2 * a2 * a3)
    s3_options = [np.sqrt(1 - c3**2), -np.sqrt(1 - c3**2)]  # elbow-up, elbow-down

    theta2_options = [np.rad2deg(np.arctan2(s, r) - np.arctan2(a3*s3, a2 + a3*c3)) for s3 in s3_options]
    theta3_options = [np.rad2deg(np.arctan2(s3, c3)) for s3 in s3_options]
    alfa = np.rad2deg(np.arcsin(x_04))
    theta4_options = [alfa - t2 - t3 for t2, t3 in zip(theta2_options, theta3_options)]

    # Create candidate solutions relative to servo zero
    def to_relative(theta_deg, servo_id):
        return theta_deg - MOTOR_LIMITS[servo_id]["deg_zero"]

    solA_rel = [to_relative(theta1_deg, 1),
                to_relative(theta2_options[0], 2),
                to_relative(theta3_options[0], 3),
                to_relative(theta4_options[0], 4)]  # elbow-up

    solB_rel = [to_relative(theta1_deg, 1),
                to_relative(theta2_options[1], 2),
                to_relative(theta3_options[1], 3),
                to_relative(theta4_options[1], 4)]  # elbow-down

    # ------------------ validate with limits (relative) ------------------
    def is_valid(sol):
        for i, ang in enumerate(sol, start=1):
            lo = SERVO_LIMITS_DEG[i][0] - MOTOR_LIMITS[i]["deg_zero"]
            hi = SERVO_LIMITS_DEG[i][1] - MOTOR_LIMITS[i]["deg_zero"]
            if not (lo <= ang <= hi):
                return False
        return True

    # Prefer elbow-up
    if is_valid(solA_rel):
        chosen = solA_rel
    elif is_valid(solB_rel):
        chosen = solB_rel
    else:
        # Clamp elbow-up if no solution valid
        chosen = [limit_angle(solA_rel[i], i+1) for i in range(4)]

    # Convert to radians
    return np.deg2rad(chosen)

