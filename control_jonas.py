import dynamixel_sdk as dxl
import kinematics_jonas
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
DEVICENAME = '/dev/ttyACM0'
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

def deg2dxl(servo,deg,opts=[150,58,150,150]):
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

def set_angles(q):
    for id in range(len(q)):
        if deg2dxl(id+1,np.rad2deg(q[id]))>=50 and deg2dxl(id+1,np.rad2deg(q[id]))<=950:
            packetHandler.write2ByteTxRx(portHandler, id+1, ADDR_MX_GOAL_POSITION, deg2dxl(id+1,np.rad2deg(q[id])))
        else:
            print("Angle out of range for servo ", id+1,": ", deg2dxl(id+1,np.rad2deg(q[id])))

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


if __name__ == "__main__":
    img_id = 0
    # Test the functions
    cap = cv2.VideoCapture(2)
    portHandler, packetHandler = setup_motors()
    T_03,T_04, T_05 = kinematics_jonas.forwards_kinematics(np.deg2rad(150-150), np.deg2rad(150-58), np.deg2rad(70-150), np.deg2rad(75-150))
    q = kinematics_jonas.inverseKinematics(T_04[0:3,0],T_04[0:3,3])
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
    T_03_1,T_04_1, T_05_1 = kinematics_jonas.forwards_kinematics(np.deg2rad(100-150), np.deg2rad(148-58), np.deg2rad(70-150), np.deg2rad(80-150))
    q_1 = kinematics_jonas.inverseKinematics(T_04_1[0:3,0],T_04_1[0:3,3])
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


    '''
    for i in range(36):
        q = kinematics.inverseKinematics([0,0,0],calculate_circle_step(2*np.pi/36*i))
        print(np.rad2deg(q[0]), np.rad2deg(q[1][0]), np.rad2deg(q[2][0]), np.rad2deg(q[3][0]))
        print(deg2dxl(1,np.rad2deg(q[0])), deg2dxl(2,np.rad2deg(q[1][0])), deg2dxl(3,np.rad2deg(q[2][0])), deg2dxl(4,np.rad2deg(q[3][0])))
        #set_angles([q[0]), deg2dxl(2,np.rad2deg(q[1][0])), deg2dxl(3,np.rad2deg(q[2][0])), deg2dxl(4,np.rad2deg(q[3][0]))])
        set_angles([q[0],q[1][1],q[2][1],q[3][1]])
        sleep(0.2)
    '''
    #sleep(2)
    #set_angles([q[0],q[1][1],q[2][1],q[3][1]])
    #print("q = ", np.rad2deg(q[0]), np.rad2deg(q[1][0]), np.rad2deg(q[2][0]), np.rad2deg(q[3][0]))
    #packetHandler.write2ByteTxRx(portHandler, 1, ADDR_MX_GOAL_POSITION, deg2dxl(1,np.rad2deg(q[0])))
    #packetHandler.write2ByteTxRx(portHandler, 2, ADDR_MX_GOAL_POSITION, deg2dxl(2,np.rad2deg(q[1][0])))
    #packetHandler.write2ByteTxRx(portHandler, 3, ADDR_MX_GOAL_POSITION, deg2dxl(3,np.rad2deg(q[2][0])))
    #packetHandler.write2ByteTxRx(portHandler, 4, ADDR_MX_GOAL_POSITION, deg2dxl(4,np.rad2deg(q[3][0])))

    #print(deg2dxl(4,np.rad2deg(q[3][0])))