import numpy as np
import kinematics
import control
from time import sleep
import main


def get_relative_transform(T1, T2):
    return np.linalg.inv(T1) @ T2


def decompose_transform(T):
    R = T[:3,:3]
    t = T[:3,3]
    return R, t


def apply_transform(T, T_rel):
    return T @ T_rel


def move_to_angles(portHandler, packetHandler, theta_deg, sleep_time=1.0):
    """
    Muove il braccio agli angoli specificati (deg) e ritorna FK
    Stampa anche gli angoli reali letti dai motori
    """
    theta_rad = np.deg2rad(theta_deg)
    _, T04, T05 = kinematics.forwards_kinematics(*theta_rad)
    q_up, q_down = kinematics.inverseKinematics_position(T04[:3,3], return_both=True)
    
    # Muove sempre con soluzione Elbow Down
    control.set_angles(portHandler, packetHandler, q_down)
    sleep(sleep_time)
    
    # Legge angoli reali dai motori
    real_angles = get_current_angles(portHandler, packetHandler)
    print(f"Target angles: {theta_deg}")
    print(f"Current angles: {real_angles}\n")
    
    return T05


def replicate_back_and_forth(portHandler, packetHandler, theta_start, theta_end, theta_third, repetitions=5):
    # 1. Muovi a posizione di partenza
    T_start = move_to_angles(portHandler, packetHandler, theta_start)
    
    # 2. Muovi a posizione di arrivo
    T_end = move_to_angles(portHandler, packetHandler, theta_end)
    
    # 3. Calcola trasformazione relativa tra start e end
    T_rel = get_relative_transform(T_start, T_end)
    
    # 4. Muovi a posizione terza
    T_third = move_to_angles(portHandler, packetHandler, theta_third)
    
    # 5. Calcola posizione 4 applicando trasformazione
    T_fourth = apply_transform(T_third, T_rel)
    t4 = T_fourth[:3,3]
    
    print("Inizio movimento avanti-indietro tra pos 3 e pos 4...\n")
    
    for i in range(repetitions):
        print(f"Repetition {i+1}: 3 -> 4")
        q_up, q_down = kinematics.inverseKinematics_position(t4, return_both=True)
        control.set_angles(portHandler, packetHandler, q_down)
        sleep(1.5)
        real_angles = get_current_angles(portHandler, packetHandler)
        print(f"Current angles at pos 4: {real_angles}\n")
        
        print(f"Repetition {i+1}: 4 -> 3")
        q_up, q_down = kinematics.inverseKinematics_position(T_third[:3,3], return_both=True)
        control.set_angles(portHandler, packetHandler, q_down)
        sleep(1.5)
        real_angles = get_current_angles(portHandler, packetHandler)
        print(f"Current angles at pos 3: {real_angles}\n")
    
    print("Movimento avanti-indietro completato.")


def get_current_absolute_angles(portHandler, packetHandler):
    angles_abs_deg = []
    ticks = []
    for servo in control.DXL_IDS:
        tick, _, _ = packetHandler.read2ByteTxRx(portHandler, servo, control.ADDR_MX_PRESENT_POSITION)
        lim = control.MOTOR_LIMITS[servo]
        # tick → gradi assoluti (senza togliere deg_zero)
        deg_abs = lim["deg_min"] + (tick - lim["tick_min"]) * (lim["deg_max"] - lim["deg_min"]) / (lim["tick_max"] - lim["tick_min"])
        angles_abs_deg.append(deg_abs)
        ticks.append(tick)
    return ticks, angles_abs_deg


def get_current_angles(portHandler, packetHandler):
    angles_deg = []
    for servo in control.DXL_IDS:
        tick, _, _ = packetHandler.read2ByteTxRx(portHandler, servo, control.ADDR_MX_PRESENT_POSITION)
        lim = control.MOTOR_LIMITS[servo]
        deg = lim["deg_min"] + (tick - lim["tick_min"]) * (lim["deg_max"] - lim["deg_min"]) / (lim["tick_max"] - lim["tick_min"])
        angles_deg.append(deg - lim["deg_zero"])
    return angles_deg


if __name__ == "__main__":
    portHandler, packetHandler = control.connect()
    control.setup_motors(portHandler, packetHandler)
    
    # Porta il robot in home e stampa gli angoli reali
    main.go_home(portHandler, packetHandler, elbow="up")
    sleep(1)
    home_angles = get_current_angles(portHandler, packetHandler)
    print(f"Home angles: {home_angles}\n")
    
    theta_start = [0, 70, 30, 0]
    sleep(1)
    theta_end   = [-20, 70, 30, 0]
    sleep(1)
    theta_third = [0, 30, 0, 0]

    replicate_back_and_forth(portHandler, packetHandler, theta_start, theta_end, theta_third, repetitions=3)
