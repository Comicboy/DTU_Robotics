import cv2
import numpy as np
import kinematics
from time import sleep
import control

if __name__ == "__main__":

    # Open camera
    cap = cv2.VideoCapture(1)

    # Connect motors
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    # Move to scanning pose (camera looking down)
    control.go_home(portHandler, packetHandler, [0, 124, -83, -105])
    sleep(1.5)

    # Read angles again and update T05
    real_angles = control.get_current_angles(portHandler, packetHandler)
    real_angles_rad = np.deg2rad(real_angles)
    print(f"Real angles (rad): {np.round(real_angles_rad,3)}")

    T03, T04, T05 = kinematics.forwards_kinematics(*real_angles_rad)
    print(f"T05 (end-effector) = \n{np.round(T05,3)}")

    # -------------------------------
    # Take a picture from the camera
    # -------------------------------
    ret, frame = cap.read()

    # --------------------------------------------
    # Try detecting the circle in the world frame
    # --------------------------------------------
    X_plane = None
    X_plane, img_out = control.detect_circle_world_tilt(frame, T05)

    # If the circle is not found, move the robot and try again
    i = 0
    while X_plane is None:
        print("Could not find circle, trying again...")
        print("Current T04:\n", T04)

        # Compute next search transformation
        T = control.calculate_circle_step(i)

        # Move robot to next scanning position
        T05 = control.move_to_position(portHandler, packetHandler, T)

        # Capture a new image
        ret, frame = cap.read()

        # Try detection again
        X_plane, img_out = control.detect_circle_world_tilt(frame, T05)
        i += 4

    # --------------------------------------------
    # Show detection result on screen
    # --------------------------------------------
    cv2.imshow("Circle Detection", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------------------------
    # Move robot to the detected circle position
    # --------------------------------------------
    X_plane = [X_plane[0], X_plane[1], X_plane[2] + 15]

    T05_Final = control.move_to_position(portHandler, packetHandler, X_plane)


