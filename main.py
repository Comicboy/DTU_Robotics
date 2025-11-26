import cv2
import numpy as np
import kinematics
import wait
from time import sleep
import control


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


# ----------------------------- MAIN PROGRAMS -----------------------------

##### MAIN SINGLE POSITION
if __name__ == "__main__":
    
    print_angle_legend()

    # Connect to motors
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    control.go_home(portHandler, packetHandler)

    sleep(5)
#
    # Move to a predefined joint configuration
    pos = [100,0,0]  # joint angles in degrees
    control.move_to_position(portHandler, packetHandler, pos)
    sleep(5)
    pos = [100,50,100]  # joint angles in degrees
    control.move_to_position(portHandler, packetHandler, pos)
    sleep(5)
    pos = [100,-50,0]  # joint angles in degrees
    control.move_to_position(portHandler, packetHandler, pos)
    sleep(5)
    pos = [80,-50,-30]  # joint angles in degrees
    control.move_to_position(portHandler, packetHandler, pos)
    sleep(5)
#
   
    print("DONE")




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

