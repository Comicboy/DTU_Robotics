import cv2
import numpy as np
import kinematics
import wait
from time import sleep
import control




def print_angle_legend():
    print("\n--- Servo Angle Legend (relative to zero) ---")
    for servo_id, lim in control.MOTOR_LIMITS.items():
        print(f"Servo {servo_id}: {lim['deg_min']-lim['deg_zero']:+.1f}   0   {lim['deg_max']-lim['deg_zero']:+.1f}")
    print("---------------------------------------------\n")


def move_to_position(portHandler, packetHandler, pos):
    q_up, q_down = kinematics.inverseKinematics_position(pos, return_both=True)

    valid_up = control.ik_solution_valid(q_up)
    valid_down = control.ik_solution_valid(q_down)

    if valid_up:
        q_rad = q_up
        sol = "ELBOW UP"
    elif valid_down:
        q_rad = q_down
        sol = "ELBOW DOWN"
    else:
        print("IK failed for:", pos)
        return False

    q_deg_int = np.round(np.rad2deg(q_rad)).astype(int)
    print(f"{sol}: {q_deg_int}")

    control.set_angles(portHandler, packetHandler, np.deg2rad(q_deg_int))
    sleep(0.03)
    return True


def move_to_logical_angles(portHandler, packetHandler, theta_logical_deg):
    theta_logical_rad = np.deg2rad(theta_logical_deg)

    _, T04, _ = kinematics.forwards_kinematics(*theta_logical_rad)

    q_up, q_down = kinematics.inverseKinematics(T04[:3,0], T04[:3,3], return_both=True)

    valid_up = control.ik_solution_valid(q_up)
    valid_down = control.ik_solution_valid(q_down)

    if valid_up:
        q_rad = q_up
        sol = "ELBOW UP"
    elif valid_down:
        q_rad = q_down
        sol = "ELBOW DOWN"
    else:
        print("\n IK ERROR: both solutions out of servo limits\n")
        return

    q_deg_int = np.round(np.rad2deg(q_rad)).astype(int)
    print(f"IK solution {sol}: {q_deg_int}")

    q_rad_int = np.deg2rad(q_deg_int)

    control.set_angles(portHandler, packetHandler, q_rad_int)

    sleep(2)

def capture_image(cap, img_id):
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    fname = f"img_{img_id:03d}.jpg"
    cv2.imwrite(fname, frame)
    print("Saved", fname)
    return img_id + 1


def draw_circle_motion(portHandler, packetHandler, steps=50):
    print("\n--- Drawing circular trajectory ---")

    for i in range(steps):
        phi = 2*np.pi * (i / steps)
        pos = control.circle_point(phi)

        ok = move_to_position(portHandler, packetHandler, pos)
        if not ok:
            print("Stopping – unreachable point.\n")
            break

    print("--- Circle finished ---\n")



###### MAIN CIRCLE
if __name__ == "__main__":
    print("\n--- INIT ---\n")

    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    # Start circle
    draw_circle_motion(portHandler, packetHandler, steps=250)

    print("DONE")


##### MAIN SINGLE POSITION

#if __name__ == "__main__":
#    img_id = 0
#
#    print_angle_legend()
#
#    cap = cv2.VideoCapture(1)
#    portHandler, packetHandler = control.connect()
#    control.setup_motors(portHandler, packetHandler)
#
#    theta_logical_0 = [20, 50, 30, 0]
#    move_to_logical_angles(portHandler, packetHandler, theta_logical_0)
#    img_id = capture_image(cap, img_id)
#
#    print("DONE")
