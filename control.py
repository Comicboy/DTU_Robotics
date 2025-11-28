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
# Defines each motor’s zero position, min/max degrees, and corresponding tick values
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
    Raises RuntimeError if the port cannot be opened or if motors do not respond.
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
    Enable torque and set the default speed for all motors.
    """
    for ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write2ByteTxRx(portHandler, ID, ADDR_MX_MOVING_SPEED, 100)

# ----------------------------- DEG → TICKS -----------------------------
def deg2dxl(servo, deg_relative):
    """
    Convert a relative angle (degrees with respect to the motor zero) 
    into the corresponding Dynamixel tick value. 
    Limits are automatically applied.
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
    Instrumented: prints sent ticks for debugging.
    q_rad is expected to be relative angles (deg relative to deg_zero) in radians.
    """
    print("\n[DEBUG] set_angles called with (deg rel):", np.round(np.rad2deg(q_rad), 3))
    for i, angle in enumerate(q_rad):
        servo = i + 1
        deg_rel = np.rad2deg(angle)
        tick = deg2dxl(servo, deg_rel)
        print(f"  -> Servo {servo}: deg_rel={deg_rel:.2f} -> tick_sent={tick}")
        packetHandler.write2ByteTxRx(portHandler, servo, ADDR_MX_GOAL_POSITION, int(tick))


def get_current_angles(portHandler, packetHandler):
    """
    Read the current motor angles in degrees (relative to motor zero).
    Rounded to the nearest integer.
    Instrumented: prints read ticks and computed angles.
    """
    angles_deg = []
    for servo in DXL_IDS:
        tick_read, comm, err = packetHandler.read2ByteTxRx(portHandler, servo, ADDR_MX_PRESENT_POSITION)
        lim = MOTOR_LIMITS[servo]
        try:
            tick_read = int(tick_read)
        except:
            print(f"[ERROR] bad tick read for servo {servo}: {tick_read}")
            tick_read = 0

        deg_abs = lim["deg_min"] + (tick_read - lim["tick_min"]) * \
                  (lim["deg_max"] - lim["deg_min"]) / (lim["tick_max"] - lim["tick_min"])

        deg_rel = deg_abs - lim["deg_zero"]
        print(f"  <- Servo {servo}: tick_read={tick_read}, deg_abs={deg_abs:.2f}, deg_rel={deg_rel:.2f}")
        angles_deg.append(round(deg_rel))
    return angles_deg



def move_to_angles(portHandler, packetHandler, theta_deg, sleep_time=1.0, poll=False):
    """
    Move the robot to the specified joint angles (degrees), respecting motor limits.
    Instrumented: prints sent and read ticks.
    If poll=True, repeatedly reads the motors until they reach the target ticks.
    """
    print("\n[DEBUG] move_to_angles: requested (deg rel) =", theta_deg)
    theta_rad = np.deg2rad(theta_deg)

    set_angles(portHandler, packetHandler, theta_rad)

    if poll:
        desired_ticks = []
        for i, deg_rel in enumerate(theta_deg):
            servo = i + 1
            desired_ticks.append(deg2dxl(servo, deg_rel))

        import time
        t0 = time.time()
        timeout = max(2.0, abs(max(theta_deg) - min(theta_deg))/10.0 + 1.0)

        while True:
            ticks_read = []
            for servo in DXL_IDS:
                tick_read, _, _ = packetHandler.read2ByteTxRx(portHandler, servo, ADDR_MX_PRESENT_POSITION)
                ticks_read.append(int(tick_read))

            diffs = [abs(ticks_read[i] - desired_ticks[i]) for i in range(len(DXL_IDS))]
            print(f"[POLL] ticks_read={ticks_read}, desired={desired_ticks}, diffs={diffs}")

            if all(d <= 5 for d in diffs):
                break
            if time.time() - t0 > timeout:
                print("[POLL] timeout waiting for motors to reach target")
                break
            time.sleep(0.08)
    else:
        sleep(sleep_time)

    real_angles_deg = get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles_deg)

    _, T04, T05 = kinematics.forwards_kinematics(*real_angles_rad)
    ee_pos = T04[:3, 3]

    print(f"[RESULT] Target angles (deg): {theta_deg}")
    print(f"[RESULT] Real angles  (deg): {real_angles_deg}")
    print(f"[RESULT] End-effector position (mm): {np.round(ee_pos, 3)}\n")

    return T05

# ----------------------------- GO HOME -----------------------------
def go_home(portHandler, packetHandler, home_angles_deg = [0, 60, -50, -110], sleep_time=1.5):
    """
    Move the robot to a predefined HOME configuration (degrees).
    Does not use IK. Calculates FK using real angles and prints the end-effector position.
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
    Move the robot down to a given Cartesian position, elbow-up only.
    """
    print(f"\n--- Moving down to {pos} ---")

    q = kinematics.inverseKinematics(pos)
    print("IK solution (rad):", np.round(q,3))
    print("IK solution (deg):", np.round(np.rad2deg(q),1))

    set_angles(portHandler, packetHandler, q)
    sleep(1.5)

    real_angles = get_current_angles(portHandler, packetHandler)
    print("Real joint angles (deg):", real_angles)

    _, T04, T05 = kinematics.forwards_kinematics(*np.deg2rad(real_angles))
    print("Achieved EE position:", np.round(T04[:3,3],1))

    return T05


def calculate_circle_step(i):
    p_c = np.array([120, 0, 65])  # Circle center
    radius = 70  # Circle radius
    rot = np.array([np.cos(2*np.pi/36*i),np.sin(2*np.pi/36*i),0])
    return p_c + radius * rot




def detect_circle_world_tilt(img, T05, Z_plane=55):
    import cv2
    import numpy as np

    # --- Camera intrinsics ---
    K = np.array([[656.3658228, 0, 310.42670403],
                  [0, 657.00426074, 243.34985795],
                  [0, 0, 1]])
    dist = np.array([0.14093633, -0.30100884, -0.00250804, 0.00459299, -0.30962826])

    # --- Undistort image ---
    img_undist = cv2.undistort(img, K, dist)
    gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # --- Detect circle ---
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=20, minRadius=5, maxRadius=200)
    if circles is None:
        return None, img_undist

    circles = np.uint16(np.around(circles))
    u, v, r = circles[0][0]

    # --- Transformation from OpenCV optical frame to mounted frame 5 ---
    # OpenCV optical frame: x→right, y→down, z→forward
    # Mounted frame 5: x→forward, y→up, z→right
    R_optical_to_5 = np.array([[0, 0, 1],
                               [0, -1, 0],
                               [1, 0, 0]])

    # --- Camera pose in base frame ---
    R_cam_in_base = T05[:3, :3] @ R_optical_to_5
    p_cam_base = T05[:3, 3].astype(float)

    # --- Ray from pixel in camera frame using intrinsics ---
    pixel_h = np.array([u, v, 1.0])
    ray_cam = np.linalg.inv(K) @ pixel_h
    ray_cam /= np.linalg.norm(ray_cam)

    # --- Ray in base frame ---
    ray_base = R_cam_in_base @ ray_cam
    ray_base /= np.linalg.norm(ray_base)

    # --- Intersection with table plane ---
    # Z_plane = table height relative to base (e.g., below base → negative)
    z_table = -Z_plane
    denom = ray_base[2]
    if abs(denom) < 1e-6:
        print("Warning: ray almost parallel to the plane")
        t = 0
    else:
        t = (z_table - p_cam_base[2]) / denom

    # --- Point on the plane in base frame ---
    X_plane = p_cam_base + t * ray_base

    # --- dx, dy, dz in mounted camera frame (x forward, y up, z right) ---
    R_base_to_cam = np.linalg.inv(T05[:3, :3])
    X_cam = R_base_to_cam @ (X_plane - p_cam_base)
    dx, dy, dz = X_cam

    # --- Draw circle on image ---
    cv2.circle(img_undist, (u, v), r, (0, 255, 0), 2)
    cv2.circle(img_undist, (u, v), 2, (0, 0, 255), 3)

    # --- Debug prints ---
    print("\nPixel (u,v):", u, v)
    print("Ray base:", ray_base)
    print("t (ray length toward plane):", t)
    print("X_plane (base frame):", X_plane)
    print("dx, dy, dz (mounted camera frame):", dx, dy, dz)

    return X_plane, img_undist
