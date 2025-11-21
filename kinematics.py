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




import numpy as np

#def inverseKinematics(pos, x_dir_z):
#    """
#    4-DOF IK: usa la tua parte di calcolo del wrist center
#    ma calcola gli angoli come nel collega.
#    
#    Parameters
#    ----------
#    pos : array-like, shape (3,)
#        Desired end-effector position [x, y, z].
#    x_dir_z : float
#        z-component of the end-effector's x-axis (orientation along stylus).
#    
#    Returns
#    -------
#    q : list of 4 elements
#        Joint angles [q0, q1, q2, q3] in radians.
#    """
#    # --- Parametri robot ---
#    a2, a3, a4, d1 = 93, 93, 50, 50
#    o_04 = np.array(pos)
#    x_04 = x_dir_z
#
#    # --- Calcolo tuo centro polso ---
#    d_4z = a4 * x_04
#    u = np.sqrt(a4**2 - d_4z**2)
#
#    # Angolo base preliminare
#    theta_base = np.arctan2(o_04[1], o_04[0])
#
#    # Proiezione polso
#    x43 = u * np.cos(theta_base)
#    y43 = u * np.sin(theta_base)
#
#    # Centro del polso
#    xc = o_04[0] - x43
#    yc = o_04[1] - y43
#    zc = o_04[2] - d_4z
#
#    # --- Angoli come nel collega ---
#    q0 = np.arctan2(yc, xc)
#    r_sq = xc**2 + yc**2
#    s = zc - d1
#
#    c2 = np.round((r_sq + s**2 - a2**2 - a3**2) / (2 * a2 * a3), 4)
#    s3_options = [np.sqrt(1 - c2**2), -np.sqrt(1 - c2**2)]
#
#    q2 = [np.arctan2(s3, c2) for s3 in s3_options]
#    q1 = [np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*np.sin(q2[0]), a2 + a3*c2),
#          np.arctan2(s, np.sqrt(r_sq)) - np.arctan2(a3*np.sin(q2[1]), a2 + a3*c2)]
#    q3 = [np.arctan2(x_04, np.sqrt(x_04**2 + 0**2)) - q1[0] - q2[0],  # x_dir_z usato per polso
#          np.arctan2(x_04, np.sqrt(x_04**2 + 0**2)) - q1[1] - q2[1]]
#
#    # Preferiamo soluzione elbow-up
#    return [q0, q1[0], q2[0], q3[0]]


def inverseKinematics(x,o):
    d4=50
    x_c=np.round(o[0]-x[0]*d4,4)
    y_c=np.round(o[1]-x[1]*d4,4)
    z_c=np.round(o[2]-x[2]*d4,4)
    d1=50
    
    q0 = np.arctan2(y_c,x_c)
    r_sq=x_c**2+y_c**2
    s=z_c-d1
    c2=np.round((r_sq+s*s-93*93-93*93)/(2*93*93),4)
    
    q2 = [np.atan2(np.sqrt(1-c2*c2),c2),
          np.atan2(-np.sqrt(1-c2*c2),c2)]
    
    q1 = [np.atan2(s,np.sqrt(r_sq))-np.atan2(93*np.sin(q2[0]),
          93+93*c2),np.atan2(s,np.sqrt(r_sq))-np.atan2(93*np.sin(q2[1]),93+93*c2)]
    
    q3 = [np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))-q1[0]-q2[0],
          np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))-q1[1]-q2[1]]
    
    return [q0,q1,q2,q3]