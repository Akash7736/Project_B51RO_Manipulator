import ik
from control import * 
import numpy as np
import time
dt = 0.1

JOINT_LIMITS = [
    (-np.pi,     np.pi),
    (-np.pi/2,   np.pi/2),
    (-np.pi/2,   np.pi/2),
]

def is_reachable(target_pos, tolerance=0.03):
    pts = []
    for t1 in np.linspace(-np.pi, np.pi, 20):
        for t2 in np.linspace(-np.pi, np.pi, 20):
            for t3 in np.linspace(-np.pi, np.pi, 20):
                pts.append(ik.fk([t1, t2, t3]))
    pts = np.array(pts)
    dist = np.linalg.norm(pts - np.array(target_pos), axis=1).min()
    return dist < tolerance, dist

def rad_to_deg(rad):
    return rad * 180 / np.pi



def pick_and_place(ser, target_position_pick, target_position_place):
    # Implement the pick and place logic here
    solved, hist, ok = ik.inverse_kinematics(
        target_pos=target_position_pick,
        theta_init=None,
        max_iter=25000,
        alpha=0.005,
        tol=1e-5,
        joint_limits=JOINT_LIMITS,
    )
    if not ok:
        print("Failed to solve IK for pick position")
        return False
    pick_angles = np.array(solved)
    print(f"Pick angles (degrees): {rad_to_deg(pick_angles)}")
    # Move to pick position
    T_move = 10.0
    q_start = np.zeros(3)  # start at home position (all angles = 0)
    q_end   = pick_angles
    positions, velocities, _ = ik.quintic_trajectory(q_start, q_end, T_move, dt)
    for pos in positions:
        ctrl_cmd = np.zeros(4)
        ctrl_cmd[:3] = rad_to_deg(pos)  # Convert angles to degrees
        ctrl_cmd[3] = 1  # Open gripper
        send_command(ser, *ctrl_cmd)
        time.sleep(dt)
    # Close gripper to pick object
    send_command(ser, *[rad_to_deg(pick_angles[0]), rad_to_deg(pick_angles[1]), rad_to_deg(pick_angles[2]), 0])
    time.sleep(1)  # Wait for gripper to close
    # Move to place position
    solved_place, hist_place, ok_place = ik.inverse_kinematics(
        target_pos=target_position_place,
        theta_init=None,
        max_iter=25000,
        alpha=0.005,
        tol=1e-5,
        joint_limits=JOINT_LIMITS,
    )
    if not ok_place:
        print("Failed to solve IK for place position")
        # return False
    place_angles = np.array(solved_place)
    print(f"Place angles (degrees): {rad_to_deg(place_angles)}")
    q_start = pick_angles
    q_end   = place_angles
    positions, velocities, _ = ik.quintic_trajectory(q_start, q_end, T_move, dt)
    for pos in positions:
        ctrl_cmd = np.zeros(4)
        ctrl_cmd[:3] = rad_to_deg(pos)  # Convert angles to degrees
        ctrl_cmd[3] = 0  # Keep gripper closed
        send_command(ser, *ctrl_cmd)
        time.sleep(dt)
    # Open gripper to release object
    send_command(ser, *[rad_to_deg(place_angles[0]), rad_to_deg(place_angles[1]), rad_to_deg(place_angles[2]), 1])
    time.sleep(1)  # Wait for gripper to open
    return True


def main():
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)

    target_position_pick = (0.1, 0.1, 0.08)
    target_position_place = (0.1, -0.1, 0.08)

    success = pick_and_place(ser, target_position_pick, target_position_place)
    if success:
        print("Pick and place operation completed successfully")
    else:
        print("Pick and place operation failed")

# def main():
#     ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
#     time.sleep(2)

#     # Example target position (x, y, z) in meters
#     # target_position = (0.1, 0.1, 0.1)
#     target_position = (0.12, 0.01, 0.15)

#     # reachable, dist = is_reachable(target_position)
#     # if not reachable:
#     #     print(f"Target unreachable — closest point is {dist:.4f}m away")
#     #     return

#     solved3, hist3, ok3 = ik.inverse_kinematics(
#         target_pos=target_position,
#         theta_init=None,
#         max_iter=25000,
#         alpha=0.005,
#         tol=1e-5,
#         joint_limits=JOINT_LIMITS,
#     )
#     print(f"\nIK solved: {ok3} in {len(hist3)} iterations")
#     target_angles = np.array(solved3)
#     print(f"Target angles (radians): {target_angles}")
#     print(f"Target angles (degrees): {rad_to_deg(target_angles)}")
#     if not ok3:
#         print("IK did not converge to a solution")
#         # return
#     dt     = 0.1
#     T_move = 10.0
#     q_start = np.zeros(3)  # start at home position (all angles = 0)
#     q_end   = target_angles
#     positions, velocities, _ = ik.quintic_trajectory(q_start, q_end, T_move, dt)
#     print(f"Generated trajectory with {len(positions)} points")
#     print(f"First 5 trajectory points (radians):\n{positions[:5]}")
#     print(f"First 5 trajectory points (degrees):\n{rad_to_deg(positions[:5])}")

#     traj_idx = 0
    
#     # Calculate the joint angles using inverse kinematics
#     theta1, theta2 , theta3 = 0, 90, -90 
#     gripper_state = 0 
#     # gripper_state = 1 for open, 0 for close
#     # servo1_angle = (-90 to 90) base 
#     # sc1_angle and sc2_angle = (-90 to 90)
#     vals = np.array([0,0,0,1])  # Initialize with default values
#     ctrl_cmd = np.array([0,0,0,1])
#     send_command(ser, *vals)  # Send initial command to set gripper state
#     print("Starting trajectory execution...")

#     while True:
#         if traj_idx < len(positions):
#             vals[:3] = positions[traj_idx].copy()  # Update joint angles for current trajectory point
#             print(f"Trajectory point {traj_idx}: {positions[traj_idx]}")
#             print(f"Sending angles: {vals[:3]}")
#             ctrl_cmd[:3] = rad_to_deg(positions[traj_idx]) 
#             traj_idx += 1
    
#         else:
#             vals[:3] = q_end
#             ctrl_cmd[3] = 0
#             send_command(ser, *ctrl_cmd)
#             break  # Stop after reaching the target position and closing gripper
#         # vals = [theta1, theta2, theta3, gripper_state]
#         # ctrl_cmd[:3] = rad_to_deg(positions[traj_idx])  # Convert angles to degrees
#         print(f"Sending ctrl_cmd: {ctrl_cmd}")
#         send_command(ser, *ctrl_cmd)
#         time.sleep(dt)

#     # end_effector = ik.fk((theta1, theta2, theta3))
#     # print(f"Calculated joint angles: {theta1:.2f}, {theta2:.2f}, {theta3:.2f}")
#     # print(f"End effector position: x={end_effector[0]:.3f}, y={end_effector[1]:.3f}, z={end_effector[2]:.3f}")


#     # Send the calculated angles to the Arduino
 
  


if __name__ == "__main__":
    main()



