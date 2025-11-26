import numpy as np
import cv2

import control
from time import sleep
import threading


def get_relative_transform(T1, T2):
    return np.linalg.inv(T1) @ T2


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


def save_current_frame(filename):
    """Salva l'ultimo frame letto dalla webcam."""
    global last_frame
    if last_frame is not None:
        cv2.imwrite(filename, last_frame)
        print(f"[IMG SAVED] {filename}")
    else:
        print("[ERROR] Nessun frame disponibile da salvare.")


def replicate_back_and_forth(portHandler, packetHandler, theta_1, theta_2, repetitions=5):

    print("\n--- Moving to POSITION 1 ---")
    T1 = control.move_to_angles(portHandler, packetHandler, theta_1)
    print("\nFirst Trasformation T1:\n", T1)
    sleep(2)
    save_current_frame("pos1_before.png")

    print("\n--- Moving to POSITION 2 ---")
    T2 = control.move_to_angles(portHandler, packetHandler, theta_2)
    print("\nFirst Trasformation T2:\n", T2)
    sleep(2)
    save_current_frame("pos2_before.png")

    T_rel = get_relative_transform(T1, T2)
    print("\nRelative Transform T_rel:\n", T_rel)

    print("\n--- Starting BACK AND FORTH ---\n")

    for i in range(repetitions):
        print(f"[{i+1}] 1 → 2")
        control.move_to_angles(portHandler, packetHandler, theta_2)
        sleep(2)
        save_current_frame(f"pos2_rep_{i+1}.png")

        print(f"[{i+1}] 2 → 1")
        control.move_to_angles(portHandler, packetHandler, theta_1)
        sleep(2)
        save_current_frame(f"pos1_rep_{i+1}.png")

    print("\nCompleted.\n")
    return T_rel


if __name__ == "__main__":

    # Start webcam
    cap = cv2.VideoCapture(1)

    # Avvio thread webcam
    cam_thread = threading.Thread(target=webcam_loop, args=(cap,), daemon=True)
    cam_thread.start()

    # Connect to robot
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    sleep(2)

    # Define robot poses
    theta_1 = [0, 103, -63, -100]
    theta_2 = [0, 120, -79, -100]

    # Perform robot motion
    T_rel = replicate_back_and_forth(portHandler, packetHandler, theta_1, theta_2, repetitions=3)

    print("\nFINAL T_rel:\n", T_rel)

    cam_thread.join()
