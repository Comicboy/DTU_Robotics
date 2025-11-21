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

    # IK usando la nuova funzione combinata
    q_rad_all = kinematics.inverseKinematics(T04[:3, 0], T04[:3,3])

    # Scegli la soluzione "elbow-up" (primo elemento di ogni lista dove presente)
    q_rad = [
        q_rad_all[0],               # q0
        q_rad_all[1][0],            # q1
        q_rad_all[2][0],            # q2
        q_rad_all[3][0]             # q3
    ]

    # Converti in gradi interi
    q_deg_int = np.round(np.rad2deg(q_rad)).astype(int)
    print("IK returned angles (deg, rounded):", q_deg_int)

    # Riconverti in radianti per la funzione di controllo
    q_rad_int = np.deg2rad(q_deg_int)

    # Manda ai motori
    control.set_angles(portHandler, packetHandler, q_rad_int)

    # Aspetta che si fermi un po’ (puoi aumentare sleep se il timeout persiste)
    sleep(3)


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
    theta_logical_0 = [20, -90, 30, 0]  # gradi logici
    move_to_logical_angles(portHandler, packetHandler, theta_logical_0)
    img_id = capture_image(cap, img_id)

    # ------------------ Seconda posizione ------------------
    theta_logical_1 = [0, 70, -20, 0]
    move_to_logical_angles(portHandler, packetHandler, theta_logical_1)
    img_id = capture_image(cap, img_id)

    print("DONE")
