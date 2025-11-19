import dynamixel_sdk as dxl
import kinematics
import numpy as np
from time import sleep
import cv2
import wait

# ----------------------
# Parametri Dynamixel
# ----------------------
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_CW_COMPLIANCE_MARGIN = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE = 29
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_PUNCH = 48
PROTOCOL_VERSION = 1.0
DXL_IDS = [1,2,3,4]
DEVICENAME = 'COM3'
BAUDRATE = 1000000
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# ----------------------
# Setup Dynamixel
# ----------------------
def setup_motors():
    portHandler = dxl.PortHandler(DEVICENAME)
    packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
    portHandler.openPort()
    portHandler.setBaudRate(BAUDRATE)
    for DXL_ID in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 20)
    return portHandler, packetHandler

# ----------------------
# Conversione gradi → step Dynamixel
# ----------------------
def deg2dxl(servo,deg,opts=[150,60,133,240]):
    match servo:
        case 1:
            dxl = int((deg+opts[0])*1023/300)
        case 2:
            dxl = int((deg+opts[1])*1023/300)
        case 3:
            dxl = int((deg+opts[2])*1023/300)
        case 4:
            dxl = int((deg+opts[3])*1023/300)
    return int(dxl)

# ----------------------
# Set angles con limiti
# ----------------------
def set_angles(q):
    limits_deg = {1: (-120, 120), 2: (-90, 90), 3: (-90, 120), 4: (-60, 60)}
    for id in range(len(q)):
        servo_id = id + 1
        angle_deg = np.rad2deg(q[id])
        min_deg, max_deg = limits_deg[servo_id]
        if angle_deg < min_deg:
            angle_deg = min_deg
        elif angle_deg > max_deg:
            angle_deg = max_deg
        dxl_pos = deg2dxl(servo_id, angle_deg)
        if 50 <= dxl_pos <= 950:
            packetHandler.write2ByteTxRx(portHandler, servo_id, ADDR_MX_GOAL_POSITION, dxl_pos)

# ----------------------
# Funzioni matrici omogenee
# ----------------------
def sind(x_deg):
    return np.sin(np.deg2rad(x_deg))

def cosd(x_deg):
    return np.cos(np.deg2rad(x_deg))

def RR(i, x_deg):
    if i == 1:  # X
        A = np.array([[1,0,0],[0,cosd(x_deg),-sind(x_deg)],[0,sind(x_deg),cosd(x_deg)]])
    elif i == 2:  # Y
        A = np.array([[cosd(x_deg),0,sind(x_deg)],[0,1,0],[-sind(x_deg),0,cosd(x_deg)]])
    elif i == 3:  # Z
        A = np.array([[cosd(x_deg),-sind(x_deg),0],[sind(x_deg),cosd(x_deg),0],[0,0,1]])
    else:
        raise ValueError("Indice asse i deve essere 1,2 o 3")
    return A

def rot_tras(I, t, d):
    R = RR(I[0], t[0]) @ RR(I[1], t[1]) @ RR(I[2], t[2])
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = d
    return T

def T_from_to(T1, T2):
    return np.linalg.inv(T1) @ T2

# ----------------------
# Esempio circle step (opzionale)
# ----------------------
def calculate_circle_step(phi):
    p_c = np.array([150, 0, 120])
    radius = 32
    rot = np.array([0,np.cos(phi),np.sin(phi)])
    return p_c + radius * rot

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    
    img_id = 0
    cap = cv2.VideoCapture(1)
    portHandler, packetHandler = setup_motors()

    # --- 1. Prima posizione della camera (scelta manuale) ---
    # Angoli dei motori [rad] per la prima posizione
    T_03,T_04, T_05 = kinematics.forwards_kinematics(np.deg2rad(150-150), np.deg2rad(147-60), np.deg2rad(86-133), np.deg2rad(134-240))
    q = kinematics.inverseKinematics(T_04[0:3,0],T_04[0:3,3])

    print("Servo 4 angle (deg) =", np.rad2deg(q[3][1]))
    #rot_tras([np.array([1,2,3])])
    #q = kinematics.inverseKinematics([1,0,0], [50,0,236])
    print(np.rad2deg([q[0],q[1][1],q[2][1],q[3][1]]))
    set_angles([q[0],q[1][1],q[2][1],q[3][1]])
    wait.wait_until_stopped(packetHandler, portHandler, DXL_IDS,
                            ADDR_MX_GOAL_POSITION, ADDR_MX_PRESENT_POSITION)

    # Salva immagine dalla camera
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    cv2.imwrite(f"img_{img_id:03d}.jpg", frame)
    print("Saved", f"img_{img_id:03d}.jpg")
    img_id += 1

    # --- 2. Definisci la trasformazione relativa da applicare ---
    # Esempio: rotazione 10° attorno a Z, traslazione 20 mm in X
    T_rel = rot_tras([3, 2, 1], [0, 10, 0], [0, 0, 0])

    # --- 3. Calcola la nuova posa della camera ---
    T_05_new = T_05 @ T_rel

    # --- 4. Calcola gli angoli dei servo per la nuova posizione ---
    q_new = kinematics.inverseKinematics(T_05_new[0:3, 0], T_05_new[0:3, 3])

    # --- 5. Muovi il robot nella nuova posizione ---
    set_angles([q_new[0], q_new[1][1], q_new[2][1], q_new[3][1]])
    wait.wait_until_stopped(packetHandler, portHandler, DXL_IDS,
                            ADDR_MX_GOAL_POSITION, ADDR_MX_PRESENT_POSITION)

    # Salva immagine dalla nuova posizione
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    cv2.imwrite(f"img_{img_id:03d}.jpg", frame)
    print("Saved", f"img_{img_id:03d}.jpg")
    img_id += 1

    # --- 6. (Facoltativo) Torna alla prima posizione usando l'inversa della trasformazione ---
    T_inverse = np.linalg.inv(T_rel)
    T_05_return = T_05_new @ T_inverse
    q_return = kinematics.inverseKinematics(T_05_return[0:3, 0], T_05_return[0:3, 3])
    set_angles([q_return[0], q_return[1][1], q_return[2][1], q_return[3][1]])
    wait.wait_until_stopped(packetHandler, portHandler, DXL_IDS,
                            ADDR_MX_GOAL_POSITION, ADDR_MX_PRESENT_POSITION)

    print("Robot returned to initial position.")
