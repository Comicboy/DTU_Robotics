import dynamixel_sdk as dxl
import kinematics
import numpy as np
from time import sleep

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
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 100)
    return portHandler, packetHandler

def deg2dxl(servo,deg,opts=[150,58,133,245]):
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
        if deg2dxl(id+1,np.rad2deg(q[id]))>=0 and deg2dxl(id+1,np.rad2deg(q[id]))<=1023:
            packetHandler.write2ByteTxRx(portHandler, id+1, ADDR_MX_GOAL_POSITION, deg2dxl(id+1,np.rad2deg(q[id])))
        else:
            print("Angle out of range for servo ", id+1,": ", deg2dxl(id+1,np.rad2deg(q[id])))
    

def calculate_circle_step(phi):
    p_c = np.array([150, 0, 120])  # Center of the circle
    radius = 32  # Radius of the circle
    rot = np.array([0,np.cos(phi),np.sin(phi)])
    return p_c + radius * rot


if __name__ == "__main__":
    # Test the functions
    portHandler, packetHandler = setup_motors()
    #T_03,T_04, T_05 = kinematics.forwards_kinematics(np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), np.deg2rad(0))
    #q = kinematics.inverseKinematics(T_04[0:3,0],T_04[0:3,3])
    #q = kinematics.inverseKinematics([1,0,0], [143,0,143])
    #print(np.rad2deg([q[0],q[1][0],q[2][0],q[3][0]]))
    #set_angles([q[0],q[1][1],q[2][1],q[3][1]])

    for i in range(36):
        q = kinematics.inverseKinematics([0,0,0],calculate_circle_step(2*np.pi/36*i))
        print(np.rad2deg(q[0]), np.rad2deg(q[1][0]), np.rad2deg(q[2][0]), np.rad2deg(q[3][0]))
        print(deg2dxl(1,np.rad2deg(q[0])), deg2dxl(2,np.rad2deg(q[1][0])), deg2dxl(3,np.rad2deg(q[2][0])), deg2dxl(4,np.rad2deg(q[3][0])))
        #set_angles([q[0]), deg2dxl(2,np.rad2deg(q[1][0])), deg2dxl(3,np.rad2deg(q[2][0])), deg2dxl(4,np.rad2deg(q[3][0]))])
        set_angles([q[0],q[1][1],q[2][1],q[3][1]])
        sleep(0.2)
    #sleep(2)
    #set_angles([q[0],q[1][1],q[2][1],q[3][1]])
    #print("q = ", np.rad2deg(q[0]), np.rad2deg(q[1][0]), np.rad2deg(q[2][0]), np.rad2deg(q[3][0]))
    #packetHandler.write2ByteTxRx(portHandler, 1, ADDR_MX_GOAL_POSITION, deg2dxl(1,np.rad2deg(q[0])))
    #packetHandler.write2ByteTxRx(portHandler, 2, ADDR_MX_GOAL_POSITION, deg2dxl(2,np.rad2deg(q[1][0])))
    #packetHandler.write2ByteTxRx(portHandler, 3, ADDR_MX_GOAL_POSITION, deg2dxl(3,np.rad2deg(q[2][0])))
    #packetHandler.write2ByteTxRx(portHandler, 4, ADDR_MX_GOAL_POSITION, deg2dxl(4,np.rad2deg(q[3][0])))

    #print(deg2dxl(4,np.rad2deg(q[3][0])))