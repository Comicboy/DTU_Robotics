import dynamixel_sdk as dxl
import kinematics
import numpy as np
from time import sleep

import cv2
import wait

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

def deg2dxl(servo,deg,opts=[150,60,133,240]):
    match servo:
        case 1:
            #512=0, 240=a*90+512
            dxl = int((deg+opts[0])*1023/300)# -deg*3+opts[0]
        case 2:
            dxl = int((deg+opts[1])*1023/300)#-deg*3+opts[1]
        case 3:
            dxl = int((deg+opts[2])*1023/300)#-deg*3+opts[2]
        case 4:
            dxl = int((deg+opts[3])*1023/300)#-deg*3+opts[3]
        
    return int(dxl)

#def set_angles(q):
#    for id in range(len(q)):
#        if deg2dxl(id+1,np.rad2deg(q[id]))>=50 and deg2dxl(id+1,np.rad2deg(q[id]))<=950:
#            packetHandler.write2ByteTxRx(portHandler, id+1, ADDR_MX_GOAL_POSITION, deg2dxl(id+1,np.rad2deg(q[id])))
#        else:
#            print("Angle out of range for servo ", id+1,": ", deg2dxl(id+1,np.rad2deg(q[id])))

def set_angles(q):

    # Limiti in gradi per sicurezza (adattali se necessario)
    limits_deg = {
        1: (-120, 120),
        2: (-90, 90),
        3: (-90, 120),
        4: (-60, 60),   # <-- servo 4 soffre, quindi limiti stretti
    }

    for id in range(len(q)):
        servo_id = id + 1

        # Converto IK → gradi
        angle_deg = np.rad2deg(q[id])

        # Applico limiti software
        min_deg, max_deg = limits_deg[servo_id]
        if angle_deg < min_deg:
            print(f"[WARN] Servo {servo_id} limited: {angle_deg:.2f}° → {min_deg}°")
            angle_deg = min_deg
        elif angle_deg > max_deg:
            print(f"[WARN] Servo {servo_id} limited: {angle_deg:.2f}° → {max_deg}°")
            angle_deg = max_deg

        # Calcolo posizione Dynamixel
        dxl_pos = deg2dxl(servo_id, angle_deg)

        # Check sul mapping reale 0..1023 (con finestre 50-950)
        if 50 <= dxl_pos <= 1023:
            packetHandler.write2ByteTxRx(
                portHandler, servo_id, ADDR_MX_GOAL_POSITION, dxl_pos
            )
        else:
            print(f"Angle out of range for servo {servo_id}: {dxl_pos}")


def rot_tras(I, t, d):
# ROT_MAT crea la matrice omogenea 4x4
#   I = ordine degli assi [es. 2 1 3]
#   t = angoli in gradi corrispondenti
#   d = traslazione [dx dy dz]

    # Rotazione 3x3
    R = RR(I(1), t(1)) * RR(I(2), t(2)) * RR(I(3), t(3))

    # Matrice omogenea 4x4
    T = np.array([[R, d],  # d(:) garantisce colonna
         [0,0,0,1]])
    return T

def RR(i, x):
    match i:
        case 1:
            A = np.array([[1,0,0],
                 [0,np.cosd(x),-np.sind(x)],
                 [0,np.sind(x) ,np.cosd(x)]])
        case 2:
            A = np.array([[np.cosd(x),0,np.sind(x)],
                 [0,1,0],
                [-np.sind(x),0,np.cosd(x)]])
        case 3:
            A = np.array([[np.cosd(x),-np.sind(x),0],
                 [np.sind(x) ,np.cosd(x),0],
                 [0,0,1]])
    return A



def calculate_circle_step(phi):
    p_c = np.array([150, 0, 120])  # Center of the circle
    radius = 32  # Radius of the circle
    rot = np.array([0,np.cos(phi),np.sin(phi)])
    return p_c + radius * rot

# go from ref 1 to ref 2
def transformation_between(T_from, T_to):
    return np.linalg.inv(T_from) @ T_to


if __name__ == "__main__":
    img_id = 0
    # Test the functions
    cap = cv2.VideoCapture(1)
    portHandler, packetHandler = setup_motors()
    T_03,T_04, T_05 = kinematics.forwards_kinematics(np.deg2rad(150-150), np.deg2rad(147-60), np.deg2rad(86-133), np.deg2rad(134-240))
    q = kinematics.inverseKinematics(T_04[0:3,0],T_04[0:3,3])

    print("Servo 4 angle (deg) =", np.rad2deg(q[3][1]))
    #rot_tras([np.array([1,2,3])])
    #q = kinematics.inverseKinematics([1,0,0], [50,0,236])
    print(np.rad2deg([q[0],q[1][1],q[2][1],q[3][1]]))
    set_angles([q[0],q[1][1],q[2][1],q[3][1]])
    wait.wait_until_stopped(packetHandler, portHandler,
        ids=DXL_IDS,
        addr_goal=ADDR_MX_GOAL_POSITION,
        addr_present=ADDR_MX_PRESENT_POSITION,
        eps_deg=1.0, consecutive=3, timeout_s=10.0)
    ret = False
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    filename = f"img_{img_id:03d}.jpg"
    cv2.imwrite(filename, frame)
    print("Saved", filename)
    img_id += 1
    sleep(1)
    T_03_1,T_04_1, T_05_1 = kinematics.forwards_kinematics(np.deg2rad(150-150), np.deg2rad(149-60), np.deg2rad(79-133), np.deg2rad(138-240))
    q_1 = kinematics.inverseKinematics(T_04_1[0:3,0],T_04_1[0:3,3])
    set_angles([q_1[0],q_1[1][1],q_1[2][1],q_1[3][1]])
    wait.wait_until_stopped(packetHandler, portHandler,
        ids=DXL_IDS,
        addr_goal=ADDR_MX_GOAL_POSITION,
        addr_present=ADDR_MX_PRESENT_POSITION,
        eps_deg=1.0, consecutive=3, timeout_s=10.0)
    #cap = cv2.VideoCapture(2)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    while not ret:
        print("New frame")
        ret, frame = cap.read()
    filename = f"img_{img_id:03d}.jpg"
    cv2.imwrite(filename, frame)
    print("Saved", filename)
    img_id += 1


    T_05_to_05_1 = transformation_between(T_05, T_05_1)
    print("Transformation between the 2 position:\n", T_05_to_05_1)

    T_reconstructed = transformation_between(T_05_1, T_05)

    q_return = kinematics.inverseKinematics(T_reconstructed[0:3,0], T_reconstructed[0:3,3])
    set_angles([q_return[0], q_return[1][1], q_return[2][1], q_return[3][1]])
