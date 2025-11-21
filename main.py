import cv2
import numpy as np
import kinematics
import wait
from time import sleep

import control  # <<— import del file separato

def print_angle_legend():
    """
    Mostra intervalli relativi allo zero di ciascun servo.
    """
    print("\n--- Servo Angle Legend (relative to zero) ---")
    for servo_id, lim in control.MOTOR_LIMITS.items():
        deg_min_rel = lim["deg_min"] - lim["deg_zero"]
        deg_max_rel = lim["deg_max"] - lim["deg_zero"]
        print(f"Servo {servo_id}: {deg_min_rel:+.1f}   0   {deg_max_rel:+.1f}")
    print("---------------------------------------------\n")


if __name__ == "__main__":
    img_id = 0

    # ------------------ PRINT ANGLE LEGEND ------------------
    print_angle_legend()

    # ------------------ INIT ------------------
    cap = cv2.VideoCapture(1)
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    # ------------------ FIRST POSITION ------------------

    theta_logical = [0, 50, -40, 0]  # angoli relativi al centro

    theta_logical_rad = np.deg2rad(theta_logical)

    T03, T04, T05 = kinematics.forwards_kinematics(*theta_logical_rad)

    # IK calcola angoli logici (relativi allo zero) a partire da T04
    q_logical_rad = kinematics.inverseKinematics(T04[:3,3], 0)

    print("Angles logical (deg):", np.rad2deg(q_logical_rad))

    # set_angles converte logico → motore → tick
    control.set_angles(portHandler, packetHandler, q_logical_rad)

    wait.wait_until_stopped(
        packetHandler, portHandler,
        ids=control.DXL_IDS,
        addr_goal=control.ADDR_MX_GOAL_POSITION,
        addr_present=control.ADDR_MX_PRESENT_POSITION,
        eps_deg=1.0, consecutive=3, timeout_s=10
    )

    # ------------------ CAPTURE IMAGE ------------------
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()

    fname = f"img_{img_id:03d}.jpg"
    cv2.imwrite(fname, frame)
    print("Saved", fname)
    img_id += 1



    # ------------------ MOVE TO SECOND POINT ------------------
    theta_logical_1 = [0, 70, -20, 0]

    
    theta_logical_rad_1 = np.deg2rad(theta_logical_1)
    
    T03_1, T04_1, T05_1 = kinematics.forwards_kinematics(*theta_logical_rad_1)

    q_1 = kinematics.inverseKinematics(T04_1[:3,3], 0)
    print("Angles (deg):", np.rad2deg(q_1))

    control.set_angles(portHandler, packetHandler, q_1)

    wait.wait_until_stopped(
        packetHandler, portHandler,
        ids=control.DXL_IDS,
        addr_goal=control.ADDR_MX_GOAL_POSITION,
        addr_present=control.ADDR_MX_PRESENT_POSITION,
        eps_deg=1.0, consecutive=3, timeout_s=10
    )

    for _ in range(10):
        cap.read()
    ret, frame = cap.read()

    fname = f"img_{img_id:03d}.jpg"
    cv2.imwrite(fname, frame)
    print("Saved", fname)
    img_id += 1

    print("DONE")

