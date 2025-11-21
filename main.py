import cv2
import numpy as np
import kinematics
import wait
from time import sleep
import control

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

def move_to_logical_angles(portHandler, packetHandler, theta_logical_deg):
    """
    Muove il robot a angoli logici (deg) usando FK -> IK -> servo.
    """
    # Converti in radianti per FK
    theta_logical_rad = np.deg2rad(theta_logical_deg)

    # FK
    T03, T04, T05 = kinematics.forwards_kinematics(*theta_logical_rad)

    # IK
    q_rad = kinematics.inverseKinematics(T04[:3, 3], 0)
    print("IK returned angles (deg):", np.rad2deg(q_rad))

    # Manda ai motori
    control.set_angles(portHandler, packetHandler, q_rad)

    # Aspetta che si fermi
    wait.wait_until_stopped(
        packetHandler, portHandler,
        ids=control.DXL_IDS,
        addr_goal=control.ADDR_MX_GOAL_POSITION,
        addr_present=control.ADDR_MX_PRESENT_POSITION,
        eps_deg=1.0, consecutive=3, timeout_s=10
    )

def capture_image(cap, img_id):
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    fname = f"img_{img_id:03d}.jpg"
    cv2.imwrite(fname, frame)
    print("Saved", fname)
    return img_id + 1

if __name__ == "__main__":
    img_id = 0

    # ------------------ Stampa limiti ------------------
    print_angle_legend()

    # ------------------ Init ------------------
    cap = cv2.VideoCapture(1)
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    # ------------------ Prima posizione ------------------
    theta_logical_0 = [0, 50, 10, 0]  # gradi logici
    move_to_logical_angles(portHandler, packetHandler, theta_logical_0)
    img_id = capture_image(cap, img_id)

    # ------------------ Seconda posizione ------------------
    theta_logical_1 = [0, 70, -20, 0]
    move_to_logical_angles(portHandler, packetHandler, theta_logical_1)
    img_id = capture_image(cap, img_id)

    print("DONE")
