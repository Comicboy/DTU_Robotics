import dynamixel_sdk as dxl
import numpy as np
import kinematics
from time import sleep
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------- CONSTANTS -----------------------------
# Control table addresses and communication parameters
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_CW_COMPLIANCE_MARGIN = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE = 29
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000
TORQUE_ENABLE = 1

# List of motor IDs
DXL_IDS = [1, 2, 3, 4]

# ----------------------------- MOTOR LIMITS -----------------------------
# Defines each motor's zero position, min/max degrees, and corresponding tick values
MOTOR_LIMITS = {
    1: {"deg_min": 59.77,  "deg_zero": 150, "deg_max": 240.23,
        "tick_min": 204, "tick_zero": 512, "tick_max": 820},
    2: {"deg_min": 42.48,  "deg_zero": 59.77,  "deg_max": 205.08,  
        "tick_min": 145, "tick_zero": 204, "tick_max": 700},
    3: {"deg_min": 29.30,  "deg_zero": 150, "deg_max": 273.93,  
        "tick_min": 100, "tick_zero": 512, "tick_max": 935},
    4: {"deg_min": 41.02, "deg_zero": 150, "deg_max": 240.23,
        "tick_min": 140, "tick_zero": 512, "tick_max": 820},
}

# ----------------------------- CONNECTION -----------------------------
def connect(port="COM7"):
    """
    Connect to the Dynamixel motors via serial port.
    Returns portHandler and packetHandler for communication.
    Raises RuntimeError if port cannot be opened or motors do not respond.
    """
    portHandler = dxl.PortHandler(port)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

    print("[INFO] Opening port...")
    if not portHandler.openPort():
        raise RuntimeError("Cannot open serial port")
    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot set baudrate")

    print("[INFO] Port OK")

    for ID in DXL_IDS:
        print(f"[INFO] Pinging motor {ID}...")
        model, comm, error = packetHandler.ping(portHandler, ID)
        if comm != dxl.COMM_SUCCESS:
            raise RuntimeError(f"Motor {ID} not responding")
        print(f"   ✔ Motor {ID} detected (model {model})")

    return portHandler, packetHandler

# ----------------------------- MOTOR INITIALIZATION -----------------------------
def setup_motors(portHandler, packetHandler):
    """
    Enable torque and set default speed for all motors.
    """
    for ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write2ByteTxRx(portHandler, ID, ADDR_MX_MOVING_SPEED, 100)

# ----------------------------- DEG → TICKS -----------------------------
def deg2dxl(servo, deg_relative):
    """
    Convert a relative degree angle (with respect to motor zero) into
    the corresponding Dynamixel tick value. Limits are automatically applied.
    """
    lim = MOTOR_LIMITS[servo]
    deg_absolute = deg_relative + lim["deg_zero"]
    deg_absolute = max(lim["deg_min"], min(lim["deg_max"], deg_absolute))

    tick = lim["tick_min"] + (deg_absolute - lim["deg_min"]) * \
           (lim["tick_max"] - lim["tick_min"]) / (lim["deg_max"] - lim["deg_min"])
    return int(round(tick))

# ----------------------------- SET ANGLES -----------------------------
# --- Replace these functions in control.py with the instrumented versions below ---

def set_angles(portHandler, packetHandler, q_rad):
    """
    Send joint angles (in radians) to motors, converting them into ticks.
    Instrumented: prints ticks sent for debugging.
    q_rad expected to be relative angles (deg relative to deg_zero) in radians.
    """
    print("\n[DEBUG] set_angles called with (deg rel):", np.round(np.rad2deg(q_rad), 3))
    for i, angle in enumerate(q_rad):
        servo = i + 1
        deg_rel = np.rad2deg(angle)
        tick = deg2dxl(servo, deg_rel)
        print(f"  -> Servo {servo}: deg_rel={deg_rel:.2f} -> tick_sent={tick}")
        # write goal position
        packetHandler.write2ByteTxRx(portHandler, servo, ADDR_MX_GOAL_POSITION, int(tick))


def get_current_angles(portHandler, packetHandler):
    """
    Read the current motor angles in degrees (relative to motor zero).
    Rounded to the nearest integer.
    Instrumented: prints ticks read and computed degrees.
    """
    angles_deg = []
    for servo in DXL_IDS:
        tick_read, comm, err = packetHandler.read2ByteTxRx(portHandler, servo, ADDR_MX_PRESENT_POSITION)
        lim = MOTOR_LIMITS[servo]
        # Defensive: ensure tick_read is int
        try:
            tick_read = int(tick_read)
        except:
            print(f"[ERROR] read bad tick for servo {servo}: {tick_read}")
            tick_read = 0
        deg_abs = lim["deg_min"] + (tick_read - lim["tick_min"]) * \
                  (lim["deg_max"] - lim["deg_min"]) / (lim["tick_max"] - lim["tick_min"])
        deg_rel = deg_abs - lim["deg_zero"]
        print(f"  <- Servo {servo}: tick_read={tick_read}, deg_abs={deg_abs:.2f}, deg_rel={deg_rel:.2f}")
        angles_deg.append(round(deg_rel))
    return angles_deg


def get_current_absolute_angles(portHandler, packetHandler):
    """
    Read current motor angles in absolute degrees (without subtracting motor zero).
    Returns ticks and absolute degrees. Instrumented printing included.
    """
    angles_abs_deg = []
    ticks = []
    for servo in DXL_IDS:
        tick_read, comm, err = packetHandler.read2ByteTxRx(portHandler, servo, ADDR_MX_PRESENT_POSITION)
        try:
            tick_read = int(tick_read)
        except:
            print(f"[ERROR] read bad tick for servo {servo}: {tick_read}")
            tick_read = 0
        lim = MOTOR_LIMITS[servo]
        deg_abs = lim["deg_min"] + (tick_read - lim["tick_min"]) * (lim["deg_max"] - lim["deg_min"]) / (lim["tick_max"] - lim["tick_min"])
        angles_abs_deg.append(deg_abs)
        ticks.append(tick_read)
        print(f"  <- Servo {servo}: tick_read={tick_read}, deg_abs={deg_abs:.2f}")
    return ticks, angles_abs_deg


def move_to_angles(portHandler, packetHandler, theta_deg, sleep_time=1.0, poll=False):
    """
    Move the robot to the specified joint angles (degrees), respecting motor limits.
    Instrumented: prints ticks sent and ticks read after motion.
    If poll=True, it reads motors repeatedly until they are within tolerance of the commanded ticks.
    """
    print("\n[DEBUG] move_to_angles: requested (deg rel) =", theta_deg)
    # 1. Convert degrees to radians (these are relative degrees)
    theta_rad = np.deg2rad(theta_deg)

    # 2. Send angles to motors (this prints ticks sent)
    set_angles(portHandler, packetHandler, theta_rad)

    # 3. Optionally poll until motion done (or wait fixed time)
    if poll:
        # convert desired to ticks for comparison
        desired_ticks = []
        for i, deg_rel in enumerate(theta_deg):
            servo = i + 1
            desired_ticks.append(deg2dxl(servo, deg_rel))
        # poll loop
        import time
        t0 = time.time()
        timeout = max(2.0, abs(max(theta_deg) - min(theta_deg))/10.0 + 1.0)  # naive timeout
        while True:
            ticks_read = []
            for servo in DXL_IDS:
                tick_read, _, _ = packetHandler.read2ByteTxRx(portHandler, servo, ADDR_MX_PRESENT_POSITION)
                ticks_read.append(int(tick_read))
            diffs = [abs(ticks_read[i] - desired_ticks[i]) for i in range(len(DXL_IDS))]
            print(f"[POLL] ticks_read={ticks_read}, desired={desired_ticks}, diffs={diffs}")
            if all(d <= 5 for d in diffs):  # tolerance in ticks
                break
            if time.time() - t0 > timeout:
                print("[POLL] timeout waiting for motors to reach target")
                break
            time.sleep(0.08)
    else:
        sleep(sleep_time)

    # 4. Read actual motor angles and show FK
    real_angles_deg = get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles_deg)

    _, T04, T05 = kinematics.forwards_kinematics(*real_angles_rad)
    ee_pos = T04[:3, 3]

    print(f"[RESULT] Target angles (deg): {theta_deg}")
    print(f"[RESULT] Real angles  (deg): {real_angles_deg}")
    print(f"[RESULT] End-effector position (mm): {np.round(ee_pos, 3)}\n")

    return T05

# ----------------------------- GO HOME -----------------------------
def go_home(portHandler, packetHandler, home_angles_deg = [0, 60, -50, -100], sleep_time=3):
    """
    Move the robot to a predefined HOME position (degrees).
    Does not use IK. Computes FK using real motor angles and prints end-effector position.
    """
    
    print(f"\nMoving to HOME (deg): {home_angles_deg}")

    home_angles_rad = np.deg2rad(home_angles_deg)
    set_angles(portHandler, packetHandler, home_angles_rad)

    sleep(sleep_time)

    real_angles_deg = get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles_deg)
    _, T04, _ = kinematics.forwards_kinematics(*real_angles_rad)

    pos = T04[:3, 3]
    print(f"Real motor angles (deg): {real_angles_deg}")
    print(f"Home position (real, mm): {np.round(pos, 3)}\n")

# ----------------------------- MOVE TO POSITION -----------------------------
def move_to_position(portHandler, packetHandler, pos):
    """
    Move robot down to a given Cartesian position, elbow-up only.
    """
    print(f"\n--- Moving down to {pos} ---")

    q = kinematics.inverseKinematics(pos)
    print("IK solution (rad):", np.round(q,3))
    print("IK solution (deg):", np.round(np.rad2deg(q),1))

    set_angles(portHandler, packetHandler, q)
    sleep(4)

    real_angles = get_current_angles(portHandler, packetHandler)
    print("Real joint angles (deg):", real_angles)

    _, T04, _ = kinematics.forwards_kinematics(*np.deg2rad(real_angles))
    print("Achieved EE position:", np.round(T04[:3,3],1))






def detect_circle_world(img, T05, Z_plane = 50):
    # --- Camera intrinsics ---
    K = np.array([[656.3658228, 0, 310.42670403],
                  [0, 657.00426074, 243.34985795],
                  [0, 0, 1]])
    dist = np.array([0.14093633, -0.30100884, -0.00250804,  0.00459299, -0.30962826])

    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # --- Undistort ---
    img_undist = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=20, minRadius=5, maxRadius=200)
    if circles is None:
        return None, img_undist

    circles = np.uint16(np.around(circles))
    u, v, r = circles[0][0]

    du = u - cx
    dv = v - cy


    # --- Pixel to normalized coordinates ---
    x_n = du / fx 
    y_n = dv / fy

    
    dx = x_n * (T05[2,3] + Z_plane)
    dy = y_n * (T05[2,3] + Z_plane)
    dz = T05[2,3] + Z_plane

    print ("\ndx: ",dx)
    print ("\ndy: ",dy)
    print ("\ndz: ",dz)

    # --- Draw circle on image ---
    cv2.circle(img_undist, (u, v), r, (0,255,0), 2)
    cv2.circle(img_undist, (u, v), 2, (0,0,255), 3)
   

    return dx, dy , dz, img_undist


def detect_circle_world_tilt(img, T05, Z_plane=50):
    import cv2
    import numpy as np

    # --- Camera intrinsics ---
    K = np.array([[656.3658228, 0, 310.42670403],
                  [0, 657.00426074, 243.34985795],
                  [0, 0, 1]])
    dist = np.array([0.14093633, -0.30100884, -0.00250804, 0.00459299, -0.30962826])

    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # --- Undistort ---
    img_undist = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # --- Detect circle ---
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=20, minRadius=5, maxRadius=200)
    if circles is None:
        return None, None, None, img_undist

    circles = np.uint16(np.around(circles))
    u, v, r = circles[0][0]

    du = float(u - cx)
    dv = float(v - cy)

    # --- Pixel to normalized coordinates ---
    x_n = du / fx
    y_n = dv / fy

    # --- Frame optical -> frame 5 (montaggio) ---
    # OpenCV optical frame: x→destra, y→basso, z→avanti
    # Frame 5: x→avanti, y→alto, z→destra
    R_optical_to_5 = np.array([[0, 0, 1],
                               [0, -1, 0],
                               [1, 0, 0]])

    # --- Raggio in frame base ---
    R_cam_in_base = T05[:3, :3] @ R_optical_to_5
    p_cam_base = T05[:3, 3].astype(float)

    ray_optical = np.array([x_n, y_n, 1.0])
    ray_base = R_cam_in_base @ ray_optical
    ray_base /= np.linalg.norm(ray_base)

    # --- Intersezione con il piano del tavolo ---
    n_plane = np.array([0.0, 0.0, 1.0])
    d_plane = -Z_plane  # tavolo sotto la base

    denom = n_plane.dot(ray_base)
    min_denom = 1e-3
    if abs(denom) < min_denom:
        print("Warning: asse ottico quasi parallelo al piano, calcolo approssimato")
        denom = np.sign(denom) * min_denom

    t = - (n_plane.dot(p_cam_base) + d_plane) / denom

    # --- Punto sul piano in frame base ---
    X_plane = p_cam_base + t * ray_base

    # --- dx, dy, dz nel frame della camera montata (x avanti, y alto, z destra) ---
    # dx/dy rispetto al centro ottico in mm
    dx = x_n * t
    dy = y_n * t
    dz = t

    print("\ndx:", dx)
    print("dy:", dy)
    print("dz:", dz)
    print("X_plane (frame base):", X_plane)

    # --- Draw circle on image ---
    cv2.circle(img_undist, (u, v), r, (0, 255, 0), 2)
    cv2.circle(img_undist, (u, v), 2, (0, 0, 255), 3)

    return X_plane, img_undist

