import dynamixel_sdk as dxl
import numpy as np

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

MOTOR_LIMITS = {
    1: {"deg_min": 60,  "deg_zero": 150, "deg_max": 240,
        "tick_min": 204, "tick_zero": 512, "tick_max": 820},
    2: {"deg_min": 51,  "deg_zero": 60,  "deg_max": 150,
        "tick_min": 175, "tick_zero": 204, "tick_max": 512},
    3: {"deg_min": 40,  "deg_zero": 130, "deg_max": 220,
        "tick_min": 140, "tick_zero": 450, "tick_max": 750},
    4: {"deg_min": 135, "deg_zero": 150, "deg_max": 240,
        "tick_min": 460, "tick_zero": 512, "tick_max": 820},
}

# ----------------------------- CONNECTION -----------------------------
def connect(port="COM3"):
    portHandler = dxl.PortHandler(port)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)

    print("[INFO] Opening port...")
    if not portHandler.openPort():
        raise RuntimeError("Cannot open serial port")

    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError("Cannot setup baudrate")

    print("[INFO] Port OK")

    # Ping motors
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
        packetHandler.write1ByteTxRx(portHandler, ID, ADDR_MX_MOVING_SPEED, 20)

# ----------------------------- DEG to TICKS -----------------------------
def deg2dxl(servo, deg_zero_relative):
    lim = MOTOR_LIMITS[servo]

    # Converto da angolo relativo a angolo assoluto
    deg_absolute = deg_zero_relative + lim["deg_zero"]

    # Clamp sui limiti assoluti
    deg_absolute = max(lim["deg_min"], min(lim["deg_max"], deg_absolute))

    # Linear mapping gradi -> tick
    tick = lim["tick_min"] + (deg_absolute - lim["deg_min"]) * \
           (lim["tick_max"] - lim["tick_min"]) / (lim["deg_max"] - lim["deg_min"])
    return int(round(tick))

# ----------------------------- SET ANGLES -----------------------------
def set_angles(portHandler, packetHandler, q_rad):
    for i, angle in enumerate(q_rad):
        servo = i + 1
        deg_relative = np.rad2deg(angle)  # angolo rispetto a zero
        tick = deg2dxl(servo, deg_relative)
        packetHandler.write2ByteTxRx(portHandler, servo, ADDR_MX_GOAL_POSITION, tick)

def logical_to_motor_angles(theta_logical):
    """
    Converte angoli logici in angoli motori assoluti (deg_real) da usare nella FK.
    theta_logical: lista di angoli in gradi relativi a zero centro ([-range, 0, +range])
    ritorna: lista di angoli in gradi assoluti compatibili con FK
    """
    motor_angles = []
    for i, th in enumerate(theta_logical, start=1):
        lim = MOTOR_LIMITS[i]
        deg_real = th + lim["deg_zero"]
        # Clamp ai limiti fisici del servo
        deg_real = max(lim["deg_min"], min(lim["deg_max"], deg_real))
        motor_angles.append(deg_real)
    return motor_angles
