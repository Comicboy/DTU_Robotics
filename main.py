import cv2
import numpy as np
import kinematics
import wait
from time import sleep
import control
import couple_movement



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
    sleep(0.5)

    # Legge e stampa gli angoli reali
    real_angles = couple_movement.get_current_angles(portHandler, packetHandler)
    print(f"Current angles (from motors): {np.round(real_angles,2)}\n")
    return True


def move_to_logical_angles(portHandler, packetHandler, theta_logical_deg):
    theta_logical_rad = np.deg2rad(theta_logical_deg)

    _, T04, _ = kinematics.forwards_kinematics(*theta_logical_rad)
    q_up, q_down = kinematics.inverseKinematics(T04[:3,0], T04[:3,3], return_both=True)

    if control.ik_solution_valid(q_up):
        q_rad = q_up
        sol = "ELBOW UP"
    elif control.ik_solution_valid(q_down):
        q_rad = q_down
        sol = "ELBOW DOWN"
    else:
        print("\n IK ERROR: both solutions out of servo limits\n")
        return

    q_deg_int = np.round(np.rad2deg(q_rad)).astype(int)
    print(f"IK solution {sol}: {q_deg_int}")

    control.set_angles(portHandler, packetHandler, np.deg2rad(q_deg_int))
    sleep(0.5)

    # Legge e stampa angoli reali
    real_angles = couple_movement.get_current_angles(portHandler, packetHandler)
    print(f"Current angles (from motors): {np.round(real_angles,2)}\n")


def capture_image(cap, img_id):
    ret, frame = cap.read()
    while not ret:
        ret, frame = cap.read()
    fname = f"img_{img_id:03d}.jpg"
    cv2.imwrite(fname, frame)
    print("Saved", fname)
    return img_id + 1


def draw_circle_motion(portHandler, packetHandler, steps=5, z_fixed=50): ##### don't go down z = 50 , really low
    """
    Muove il braccio lungo un cerchio orizzontale nel piano X-Y
    mantenendo Z fisso.
    """
    print("\n--- Drawing circular trajectory (Z fixed) ---")
    
    # Parametri cerchio
    p_center = np.array([150, 0, z_fixed])  # centro del cerchio
    radius = 20                              # raggio del cerchio

    for i in range(steps):
        phi = 2*np.pi * (i / steps)
        # punto del cerchio con Z fisso
        pos = p_center + radius * np.array([np.cos(phi), np.sin(phi), 0])
        
        # calcolo IK (ritorna [Elbow Up, Elbow Down])
        q_up, q_down = kinematics.inverseKinematics_position(pos, return_both=True)
        
        # invia direttamente la soluzione Elbow Down
        control.set_angles(portHandler, packetHandler, q_down)

        sleep(0.07)  # pausa tra step per dare tempo al braccio di muoversi

    print("--- Circle finished ---\n")


def follow_vertical_trajectory(portHandler, packetHandler, x_fixed, y_fixed, z_values, elbow="up", delay=0.1):
    """
    Traiettoria verticale con polso sempre verso il basso.
    """
    print("\n--- Following vertical trajectory (stylus down) ---")
    
    x_dir = np.array([0, 0, -1])  # direzione fissa verso il basso
    
    for z in z_values:
        pos = np.array([x_fixed, y_fixed, z])
        q_up, q_down = kinematics.inverseKinematics_position(pos, x_dir=x_dir, return_both=True)
        
        if elbow.lower() == "up":
            q = q_up
        else:
            q = q_down
        
        control.set_angles(portHandler, packetHandler, q)
        sleep(delay)
    
    print("--- Trajectory finished ---\n")


def go_home(portHandler, packetHandler, elbow="down"):
    home_pos = np.array([100, 0, 100])
    x_dir = np.array([0, 0, 0])
    q_up, q_down = kinematics.inverseKinematics_position(home_pos, x_dir=x_dir, return_both=True)

    if elbow.lower() == "up":
        q_target = q_up
        sol_name = "ELBOW UP"
    else:
        q_target = q_down
        sol_name = "ELBOW DOWN"

    print(f"Moving to HOME ({sol_name}): {np.round(np.rad2deg(q_target)).astype(int)}")
    control.set_angles(portHandler, packetHandler, q_target)
    sleep(1.5)

    # Stampa angoli reali
    real_angles = couple_movement.get_current_angles(portHandler, packetHandler)
    print(f"Current angles at HOME: {np.round(real_angles,2)}\n")

    ticks, real_abs = couple_movement.get_current_absolute_angles(portHandler, packetHandler)
    print(f"Ticks: {ticks}")
    print(f"Absolute angles: {np.round(real_abs,2)}\n")







# ------------------------ MAIN ------------------------------####


###### MAIN CIRCLE
#if __name__ == "__main__":
#    print("\n--- INIT ---\n")
#    portHandler, packetHandler = control.connect()
#    control.setup_motors(portHandler, packetHandler)
#
#    # Start circle
#    draw_circle_motion(portHandler, packetHandler, steps=20)
#    print("DONE")





##### MAIN SINGLE POSITION

if __name__ == "__main__":
    img_id = 0

    print_angle_legend()

    cap = cv2.VideoCapture(1)
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)

    
    theta_logical_0 = [20, 10, 30, 0]
    move_to_logical_angles(portHandler, packetHandler, theta_logical_0)

    

    print("DONE")





##### MAIN TRAJECTORY 

#if __name__ == "__main__":
#    print("\n--- INIT ---\n")
#
#    portHandler, packetHandler = control.connect()
#    control.setup_motors(portHandler, packetHandler)
#    go_home(portHandler, packetHandler, elbow="down")
#
#    x_fixed = 150
#    y_fixed = 0
#    z_values = np.linspace(100, 10, 10)  # sali da Z=100 a Z=150 in 50 step
#
#    follow_vertical_trajectory(portHandler, packetHandler, x_fixed, y_fixed, z_values, elbow="down")
