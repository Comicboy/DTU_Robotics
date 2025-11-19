# wait.py
import time

def wait_until_stopped(packetHandler, portHandler,
                       ids,
                       addr_goal=30,            # ADDR_MX_GOAL_POSITION
                       addr_present=36,         # ADDR_MX_PRESENT_POSITION
                       addr_moving=46,          # "Moving" flag
                       eps_deg=1.0,
                       consecutive=3,
                       timeout_s=6.0,
                       poll_hz=40.0):
    """
    Block until all servos in 'ids' report Moving=0 AND |present - goal| < eps_deg (converted to ticks)
    for 'consecutive' polls in a row, or until timeout.
    """
    UNITS_PER_DEG = 1023.0 / 300.0
    eps_ticks = int(round(eps_deg * UNITS_PER_DEG))
    good_in_a_row = 0
    dt = 1.0 / max(1.0, poll_hz)
    t0 = time.time()

    while True:
        all_good = True
        last_err = 0
        last_sid = None

        for sid in ids:
            moving, _, _ = packetHandler.read1ByteTxRx(portHandler, sid, addr_moving)
            goal,   _, _ = packetHandler.read2ByteTxRx(portHandler, sid, addr_goal)
            present,_, _ = packetHandler.read2ByteTxRx(portHandler, sid, addr_present)

            err = abs(int(goal) - int(present))
            last_err, last_sid = err, sid

            if (moving != 0) or (err > eps_ticks):
                all_good = False
                break

        if all_good:
            good_in_a_row += 1
            if good_in_a_row >= consecutive:
                return True
        else:
            good_in_a_row = 0

        if (time.time() - t0) > timeout_s:
            print(f"wait_until_stopped: timeout; last err_ticks={last_err} on servo {last_sid}")
            return False

        time.sleep(dt)
