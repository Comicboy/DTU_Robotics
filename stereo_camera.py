import kinematics
import control
import numpy as np
import cv2
import os
from time import sleep

def replicate_stereo_rotation(portHandler, packetHandler, theta_center, theta1_offset=20, repetitions=3, cam=None, save_dir="stereo_images"):
    """
    Simulate stereo motion by rotating only the first joint (theta1) between right and left positions.
    Computes and prints the relative transformation T_left * inv(T_right).
    Displays camera feed in real time continuously and saves snapshots at Left and Right positions.
    """
    # Crea cartella per salvare le immagini
    os.makedirs(save_dir, exist_ok=True)
    
    # Posizioni sinistra e destra
    theta_left  = theta_center.copy()
    theta_left[0]  = theta_center[0] - theta1_offset
    
    theta_right = theta_center.copy()
    theta_right[0] = theta_center[0] + theta1_offset
    
    window_name = "Camera Feed"
    if cam:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i in range(repetitions):
        # Right -> Left
        print(f"Repetition {i+1}: Right -> Left")
        control.move_to_angles(portHandler, packetHandler, theta_left)
        sleep(0.5)
        
        # Aggiorna feed camera mentre il robot si muove
        duration = 1.5
        start_time = cv2.getTickCount()
        fps = cv2.getTickFrequency()
        frame_left = None
        while (cv2.getTickCount() - start_time) / fps < duration:
            if cam:
                ret, frame = cam.read()
                if ret:
                    frame_left = frame.copy()
                    cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if frame_left is not None:
            left_path = os.path.join(save_dir, f"left_{i+1}.png")
            cv2.imwrite(left_path, frame_left)
            print(f"Saved Left image: {left_path}")
        
        real_angles_left = control.get_current_angles(portHandler, packetHandler)
        T_03, T_04, T_05 = kinematics.forwards_kinematics(*real_angles_left)
        print(f"Real angles at Left: {real_angles_left}")
        print(f"T_left (end-effector):\n{T_05}\n")
        
        # Left -> Right
        print(f"Repetition {i+1}: Left -> Right")
        control.move_to_angles(portHandler, packetHandler, theta_right)
        sleep(0.5)
        
        start_time = cv2.getTickCount()
        frame_right = None
        while (cv2.getTickCount() - start_time) / fps < duration:
            if cam:
                ret, frame = cam.read()
                if ret:
                    frame_right = frame.copy()
                    cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if frame_right is not None:
            right_path = os.path.join(save_dir, f"right_{i+1}.png")
            cv2.imwrite(right_path, frame_right)
            print(f"Saved Right image: {right_path}")
        
        real_angles_right = control.get_current_angles(portHandler, packetHandler)
        T_03_r, T_04_r, T_right_actual = kinematics.forwards_kinematics(*real_angles_right)
        print(f"Real angles at Right: {real_angles_right}")
        print(f"T_right_actual:\n{T_right_actual}\n")
    
    print("Stereo rotation back-and-forth completed.")

if __name__ == "__main__":
    # Connessione robot
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    
    # Posizione centrale della camera
    theta_center = [0, 90, -90, -90]
    
    # Connessione camera
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        print("Errore: impossibile aprire la camera")
        cam = None
    
    try:
        replicate_stereo_rotation(portHandler, packetHandler, theta_center, theta1_offset=10, repetitions=3, cam=cam)
    finally:
        if cam:
            cam.release()
        cv2.destroyAllWindows()
