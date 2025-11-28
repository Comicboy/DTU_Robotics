import cv2
import numpy as np
import kinematics
import wait
from time import sleep
import control
import threading


# ----------------------------- UTILITY FUNCTIONS -----------------------------

def print_angle_legend():
    """
    Print a legend showing servo angles relative to their zero positions.
    Helps visualize min, zero, and max limits for each servo.
    """
    print("\n--- Servo Angle Legend (relative to zero) ---")
    for servo_id, lim in control.MOTOR_LIMITS.items():
        print(f"Servo {servo_id}: {lim['deg_min']-lim['deg_zero']:+.1f}   0   {lim['deg_max']-lim['deg_zero']:+.1f}")
    print("---------------------------------------------\n")


def draw_circle_motion(portHandler, packetHandler, steps=5, z_fixed=50):
    """
    Move the robot along a horizontal circle in the XY plane while keeping Z fixed.
    
    Parameters:
    - steps: number of points along the circle
    - z_fixed: fixed Z coordinate (height)
    """
    print("\n--- Drawing circular trajectory (Z fixed) ---")
    
    # Circle parameters
    p_center = np.array([150, 0, z_fixed])  # circle center
    radius = 20                              # radius of the circle

    for i in range(steps):
        phi = 2 * np.pi * (i / steps)
        pos = p_center + radius * np.array([np.cos(phi), np.sin(phi), 0])
        
        # Compute IK (returns elbow-up and elbow-down solutions)
        q = kinematics.inverseKinematics(pos)
        
        # Move using the elbow-down solution
        control.set_angles(portHandler, packetHandler, q)
        sleep(0.07)  # small delay to allow motion

    print("--- Circle finished ---\n")


def follow_vertical_trajectory(portHandler, packetHandler, x_fixed, y_fixed, z_values, elbow="up", delay=0.1):
    """
    Move the robot along a vertical trajectory while keeping the end-effector pointing downwards.
    
    Parameters:
    - x_fixed, y_fixed: fixed XY coordinates
    - z_values: array of Z positions to follow
    - elbow: choose "up" or "down" configuration
    - delay: time to wait between steps
    """
    print("\n--- Following vertical trajectory (stylus down) ---")
    
    x_dir = np.array([0, 0, 0])  # fixed downward direction
    
    for z in z_values:
        pos = np.array([x_fixed, y_fixed, z])
        q_up, q_down = kinematics.inverseKinematics_position(pos, x_dir=x_dir, return_both=True)
        
        q = q_up if elbow.lower() == "up" else q_down
        
        control.set_angles(portHandler, packetHandler, q)
        sleep(delay)
    
    print("--- Trajectory finished ---\n")


def move_down_to_position(portHandler, packetHandler, pos, elbow="down"):
    """
    Move the robot to a requested (x, y, z) Cartesian position,
    keeping the end-effector pointing downward.

    Parameters:
    - pos: array/list [x, y, z] in mm
    - elbow: "up" or "down" IK branch
    """

    print(f"\n--- Moving down to position {pos} ---")

    x_dir = np.array([0, 0, -1])  # always pointing downward

    # Compute IK (returns both solutions)
    q_up, q_down = kinematics.inverseKinematics_position(
        pos,
        x_dir=x_dir,
        return_both=True
    )

    # Choose the correct branch
    q = q_down if elbow.lower() == "down" else q_up

    # Move robot
    control.set_angles(portHandler, packetHandler, q)
    sleep(1.0)

    # Read real robot angles
    real_angles = control.get_current_angles(portHandler, packetHandler)
    print(f"Real joint angles (deg): {real_angles}")

    print("--- Reached target position ---\n")

def webcam_loop(cap):
    """Thread per mostrare la webcam in continuo."""
    global last_frame
    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        last_frame = frame.copy()  # salviamo l'ultimo frame disponibile
        
        cv2.imshow("Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------- MAIN PROGRAMS -----------------------------

##### MAIN SINGLE POSITION
#if __name__ == "__main__":
#    
#
#    print_angle_legend()
#
#    # Connect to motors
#    portHandler, packetHandler = control.connect()
#    control.setup_motors(portHandler, packetHandler)
#    control.go_home(portHandler, packetHandler)
#
#    sleep(5)
   
if __name__ == "__main__":
    # 1. Connetti motori
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    control

    # 2. Vai in HOME
    control.go_home(portHandler, packetHandler,[0,90,0,0])
    sleep(3)
    real_angles = control.get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles)
    print(f"Real angles (rad): {np.round(real_angles_rad,3)}")

    # 4. Forward kinematics
    T03, T04, T05 = kinematics.forwards_kinematics(*real_angles_rad)
    print(f"T04 (end-effector) = \n{np.round(T04,3)}")
    print(f"T05 (end-effector) = \n{np.round(T05,3)}")
    control.go_home(portHandler, packetHandler,[0, 124, -83, -95])
    sleep(3)
    # 3. Leggi angoli reali dei motori
    real_angles = control.get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles)
    print(f"Real angles (rad): {np.round(real_angles_rad,3)}")
#
    # 4. Forward kinematics
    T03, T04, T05 = kinematics.forwards_kinematics(*real_angles_rad)
    print(f"T05 (end-effector) = \n{np.round(T05,3)}")
#    ## 5. Scatta foto dalla webcam
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Impossibile leggere la camera!")
#    # 7. Trova cerchio nel world frame
    #dx, dy , dz , img_out = control.detect_circle_world(frame, T05)
#
    #mov = [dx,dy,dz]
    #R_align = np.array([[0,  0, 1],
    #                    [0, -1, 0],
    #                    [1,  0, 0]])
    #
    #pos_cam_frame = R_align @ mov
#
    #print("\nPos respect to camera: ", pos_cam_frame)
#
#
    #P_cam_h = np.hstack((pos_cam_frame, 1.0))
    #P_base_h = T05 @ P_cam_h
    #P_base = P_base_h[:3]
    #
    #print("Punto nel frame base:", P_base)
#
    #P_base = [P_base[0] - 20, P_base[1], P_base[2] + 20]
#
#   # # 9. Mostra immagine
    #cv2.imshow("Circle Detection", img_out)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#
    
#
    #control.move_to_position(portHandler,packetHandler,P_base) 
    # 
    # 
       
    X_plane, img_out = control.detect_circle_world_tilt(frame, T05)
     # 9. Mostra immagine
    cv2.imshow("Circle Detection", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
    X_plane = [X_plane[0],X_plane[1], X_plane[2] - 90 ]
    
    control.move_to_position(portHandler,packetHandler,X_plane)


##### MAIN CIRCLE 
#if __name__ == "__main__":
#     print("\n--- INIT ---\n")
#     portHandler, packetHandler = control.connect()
#     control.setup_motors(portHandler, packetHandler)
#
#     #control.go_home(portHandler, packetHandler)
#     sleep(1)
#     
#     # Draw circular trajectory
#     draw_circle_motion(portHandler, packetHandler, steps=20)
#     print("DONE")




##### MAIN VERTICAL TRAJECTORY 
#if __name__ == "__main__":
#     print("\n--- INIT ---\n")
#     portHandler, packetHandler = control.connect()
#     control.setup_motors(portHandler, packetHandler)
#
#     control.go_home(portHandler, packetHandler)
#     sleep(5)
#
#     x_fixed = 150
#     y_fixed = 0
#     z_values = np.linspace(150, 100, 10)  # move down from Z=100 to Z=10 in 10 steps
#
#     follow_vertical_trajectory(portHandler, packetHandler, x_fixed, y_fixed, z_values, elbow="down")

