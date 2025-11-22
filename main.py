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
        q_up, q_down = kinematics.inverseKinematics_position(pos, return_both=True)
        
        # Move using the elbow-down solution
        control.set_angles(portHandler, packetHandler, q_down)
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
    
    x_dir = np.array([0, 0, -1])  # fixed downward direction
    
    for z in z_values:
        pos = np.array([x_fixed, y_fixed, z])
        q_up, q_down = kinematics.inverseKinematics_position(pos, x_dir=x_dir, return_both=True)
        
        q = q_up if elbow.lower() == "up" else q_down
        
        control.set_angles(portHandler, packetHandler, q)
        sleep(delay)
    
    print("--- Trajectory finished ---\n")

# ----------------------------- MAIN PROGRAMS -----------------------------

##### MAIN SINGLE POSITION
if __name__ == "__main__":
    img_id = 0

    print_angle_legend()

    # Connect to motors
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    # Move to a predefined joint configuration
    theta = [20, 10, 30, 0]  # joint angles in degrees
    control.move_to_angles(portHandler, packetHandler, theta)
    
    print("DONE")




##### MAIN CIRCLE 
# if __name__ == "__main__":
#     print("\n--- INIT ---\n")
#     portHandler, packetHandler = control.connect()
#     control.setup_motors(portHandler, packetHandler)
#
#     control.go_home(portHandler, packetHandler)
#     sleep(1)
#     
#     # Draw circular trajectory
#     draw_circle_motion(portHandler, packetHandler, steps=20)
#     print("DONE")




##### MAIN VERTICAL TRAJECTORY 
# if __name__ == "__main__":
#     print("\n--- INIT ---\n")
#     portHandler, packetHandler = control.connect()
#     control.setup_motors(portHandler, packetHandler)
#
#     control.go_home(portHandler, packetHandler)
#     sleep(1)
#
#     x_fixed = 150
#     y_fixed = 0
#     z_values = np.linspace(100, 10, 10)  # move down from Z=100 to Z=10 in 10 steps
#
#     follow_vertical_trajectory(portHandler, packetHandler, x_fixed, y_fixed, z_values, elbow="down")

