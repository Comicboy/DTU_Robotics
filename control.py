import dynamixel_sdk as dxl
import numpy as np
import kinematics
from time import sleep


# ----------------------------- CONSTANTS -----------------------------
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

DXL_IDS = [1,2,3,4]

# ----------------------------- MOTOR LIMITS -----------------------------
MOTOR_LIMITS = {
    1: {"deg_min": 60,  "deg_zero": 150, "deg_max": 240,
        "tick_min": 204, "tick_zero": 512, "tick_max": 820},
    2: {"deg_min": 51,  "deg_zero": 60,  "deg_max": 150,  
        "tick_min": 175, "tick_zero": 204, "tick_max": 512},
    3: {"deg_min": 60,  "deg_zero": 150, "deg_max": 240,   #robot 4 deg_zero = 130, robot 6 deg_zero = 150, also min and max change
        "tick_min": 204, "tick_zero": 512, "tick_max": 820},
    4: {"deg_min": 135, "deg_zero": 150, "deg_max": 240,
        "tick_min": 460, "tick_zero": 512, "tick_max": 820},
}

# ----------------------------- CONNECTION -----------------------------
def connect(port="COM4"):
    portHandler = dxl.PortHandler(port)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

    print("[INFO] Opening port...")
    if not portHandler.openPort():
        raise RuntimeError("Cannot open serial port")

    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot setup baudrate")

    print("[INFO] Port OK")

    for ID in DXL_IDS:
        print(f"[INFO] Pinging motor {ID}...")
        model, comm, error = packetHandler.ping(portHandler, ID)
        if comm != dxl.COMM_SUCCESS:
            raise RuntimeError(f"Motor {ID} not responding")
        print(f"   ✔ Motor {ID} detected (model {model})")

    return portHandler, packetHandler

# ----------------------------- MOTOR INIT -----------------------------
def setup_motors(portHandler, packetHandler):
    for ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write1ByteTxRx(portHandler, ID, ADDR_MX_MOVING_SPEED, 25)

# ----------------------------- DEG → TICKS -----------------------------
def deg2dxl(servo, deg_relative):
    lim = MOTOR_LIMITS[servo]

    deg_absolute = deg_relative + lim["deg_zero"]
    deg_absolute = max(lim["deg_min"], min(lim["deg_max"], deg_absolute))

    tick = lim["tick_min"] + (deg_absolute - lim["deg_min"]) * \
           (lim["tick_max"] - lim["tick_min"]) / (lim["deg_max"] - lim["deg_min"])
    return int(round(tick))

# ----------------------------- SET ANGLES -----------------------------
def set_angles(portHandler, packetHandler, q_rad):
    for i, angle in enumerate(q_rad):
        servo = i + 1
        
        deg_rel = np.rad2deg(angle)
        tick = deg2dxl(servo, deg_rel)
        packetHandler.write2ByteTxRx(portHandler, servo, ADDR_MX_GOAL_POSITION, tick)

# ----------------------------- IK UTILS -----------------------------
def ik_rad_to_real_deg(q_rad):
    q_deg = np.rad2deg(q_rad)
    q_real = []
    for i, deg_rel in enumerate(q_deg, start=1):
        lim = MOTOR_LIMITS[i]
        q_real.append(deg_rel + lim["deg_zero"])
    return np.array(q_real)

def ik_solution_valid(q_rad):
    #q_real = ik_rad_to_real_deg(q_rad)
    #for i, real in enumerate(q_real, start=1):
    #    lim = MOTOR_LIMITS[i]
    #    if not (lim["deg_min"] <= real <= lim["deg_max"]):
    #        return False
    return True







