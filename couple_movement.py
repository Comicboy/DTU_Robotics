import numpy as np
import kinematics
import control
from time import sleep
import main


def get_relative_transform(T1, T2):
    """
    Compute the relative transformation matrix from T1 to T2.
    """
    return np.linalg.inv(T1) @ T2


def decompose_transform(T):
    """
    Decompose a homogeneous transformation matrix into rotation and translation components.
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def apply_transform(T, T_rel):
    """
    Apply a relative transformation T_rel to a base transformation T.
    Returns the resulting transformation.
    """
    return T @ T_rel


def replicate_back_and_forth(portHandler, packetHandler, theta_start, theta_end, theta_third, repetitions=5):
    """
    Move the robot back and forth between positions.
    
    Steps:
    1. Move to start position
    2. Move to end position
    3. Compute relative transformation between start and end
    4. Move to third position
    5. Apply relative transformation to get fourth position
    6. Repeat back-and-forth motion for the specified number of repetitions
    """
    
    # 1. Move to start position
    T_start = control.move_to_angles(portHandler, packetHandler, theta_start)
    
    # 2. Move to end position
    T_end = control.move_to_angles(portHandler, packetHandler, theta_end)
    
    # 3. Compute relative transformation from start to end
    T_rel = get_relative_transform(T_start, T_end)
    
    # 4. Move to third position
    T_third = control.move_to_angles(portHandler, packetHandler, theta_third)
    
    # 5. Compute fourth position by applying the relative transformation
    T_fourth = apply_transform(T_third, T_rel)
    t4 = T_fourth[:3, 3]
    
    print("Starting back-and-forth movement between positions 3 and 4...\n")
    
    for i in range(repetitions):
        # Move from position 3 to 4
        print(f"Repetition {i+1}: 3 -> 4")
        q_up, q_down = kinematics.inverseKinematics_position(t4, return_both=True)
        control.set_angles(portHandler, packetHandler, q_down)
        sleep(1.5)
        real_angles = control.get_current_angles(portHandler, packetHandler)
        print(f"Current angles at pos 4: {real_angles}\n")
        
        # Move from position 4 back to 3
        print(f"Repetition {i+1}: 4 -> 3")
        q_up, q_down = kinematics.inverseKinematics_position(T_third[:3, 3], return_both=True)
        control.set_angles(portHandler, packetHandler, q_down)
        sleep(1.5)
        real_angles = control.get_current_angles(portHandler, packetHandler)
        print(f"Current angles at pos 3: {real_angles}\n")
    
    print("Back-and-forth movement completed.")


if __name__ == "__main__":
    # Connect to the robot and initialize motors
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    
    # Move robot to home position and print real motor angles
    control.go_home(portHandler, packetHandler, elbow="up")
    sleep(1)
    home_angles = control.get_current_angles(portHandler, packetHandler)
    print(f"Home angles: {home_angles}\n")
    
    # Define target angles for the back-and-forth movement
    theta_start = [0, 70, 30, 0]
    sleep(1)
    theta_end   = [-20, 70, 30, 0]
    sleep(1)
    theta_third = [0, 30, 0, 0]

    # Execute the back-and-forth trajectory
    replicate_back_and_forth(portHandler, packetHandler, theta_start, theta_end, theta_third, repetitions=3)

