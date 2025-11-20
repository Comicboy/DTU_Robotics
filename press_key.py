# press_key.py
# Copy this to call function:
#   from press_key import press_key
#   press_key(
#       q=[q0, q1, q2, q3],          # current joint angles [rad]
#       set_angles_fn=set_angles,    # typically control.set_angles
#       packetHandler=packetHandler,
#       portHandler=portHandler,
#       dxl_ids=DXL_IDS,
#       addr_moving_speed=ADDR_MX_MOVING_SPEED,
#       press_joint_index=2,         # 2 -> q3, 3 -> q4
#       press_delta_deg=5.0,
#       speed=80
#   )
#
# This helper performs a local "press" motion: it moves one joint down by a
# small angle (in degrees) from the current pose, waits, and returns back.

import time
import numpy as np

def set_all_speeds(packetHandler, portHandler, dxl_ids, addr_moving_speed, speed):
    """Set the same moving speed for all Dynamixel motors."""
    for DXL_ID in dxl_ids:
        packetHandler.write2ByteTxRx(
            portHandler,
            DXL_ID,
            addr_moving_speed,
            int(speed)
        )

def press_key(
    q,
    set_angles_fn,               # e.g. control.set_angles
    packetHandler=None,
    portHandler=None,
    dxl_ids=None,
    addr_moving_speed=None,
    press_joint_index=2,         # 2 -> q3, 3 -> q4
    press_delta_deg=5.0,         # how much to rotate that joint (in degrees)
    speed=None,                  # if None, keep current speed
    press_wait=0.2,              # how long to stay pressed
    release_wait=0.2             # wait after releasing
):
    """
    q: list/array [q0, q1, q2, q3] in radians – the pose you are ALREADY at.
    set_angles_fn: function to send joint angles to the robot (e.g. control.set_angles)

    If 'speed' is not None, you must also pass:
      - packetHandler, portHandler, dxl_ids, addr_moving_speed
    so we can update the speeds.
    """

    # Optional: change speed before pressing
    if speed is not None:
        if packetHandler is None or portHandler is None or dxl_ids is None or addr_moving_speed is None:
            raise ValueError("To change speed, provide packetHandler, portHandler, dxl_ids, addr_moving_speed")
        set_all_speeds(packetHandler, portHandler, dxl_ids, addr_moving_speed, speed)

    base_q = list(q)  # copy of the current joint angles

    # 1) Press down: modify only the selected joint, in radians
    press_q = base_q.copy()
    press_q[press_joint_index] -= np.deg2rad(press_delta_deg)

    set_angles_fn(press_q)
    time.sleep(press_wait)

    # 2) Release: go back to the original pose
    set_angles_fn(base_q)
    time.sleep(release_wait)
