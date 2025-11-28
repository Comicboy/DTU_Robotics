# debug_position.py
# Usage:
#   from debug_position import print_desired_and_actual
#   print_desired_and_actual(
#       q=[q[0], q[1][1], q[2][1], q[3][1]],
#       packetHandler=packetHandler,
#       portHandler=portHandler,
#       DXL_IDS=DXL_IDS,
#       ADDR_MX_PRESENT_POSITION=ADDR_MX_PRESENT_POSITION,
#       deg2dxl_fn=deg2dxl,
#       label="after move"
#   )
#
# This helper prints desired joint angles, desired Dynamixel ticks and actual
# servo positions, to debug why a pose/command might not be reached.

import numpy as np

def print_desired_and_actual(
    q,
    packetHandler,
    portHandler,
    DXL_IDS,
    ADDR_MX_PRESENT_POSITION,
    deg2dxl_fn,
    label: str = ""
):
    """
    q: list/array [q0, q1, q2, q3] in radians.
    Prints desired joint angles (deg), desired ticks, and actual ticks.
    All other arguments are passed in from control.py so we avoid circular imports.
    """
    if label:
        print(f"\n=== DEBUG: {label} ===")
    else:
        print("\n=== DEBUG: desired vs actual ===")

    # Desired joint angles in degrees
    q_deg = [np.rad2deg(a) for a in q]
    print(f"Desired joint angles (deg): {q_deg}")

    # Desired Dynamixel positions (ticks)
    desired_ticks = [
        deg2dxl_fn(servo_id, q_deg[i])
        for i, servo_id in enumerate(DXL_IDS)
    ]
    print(f"Desired Dynamixel ticks: {desired_ticks}")

    # Actual Dynamixel positions (ticks) and error
    actual_ticks = []
    for i, DXL_ID in enumerate(DXL_IDS):
        present_pos, comm_result, dxl_error = packetHandler.read2ByteTxRx(
            portHandler,
            DXL_ID,
            ADDR_MX_PRESENT_POSITION
        )
        actual_ticks.append(present_pos)
        err = present_pos - desired_ticks[i]
        print(
            f"Servo {DXL_ID}: "
            f"goal={desired_ticks[i]}, "
            f"present={present_pos}, "
            f"err={err}"
        )

    print(f"Actual Dynamixel ticks: {actual_ticks}")
    print("=== END DEBUG ===\n")
