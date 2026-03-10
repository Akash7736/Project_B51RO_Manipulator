import mujoco
import numpy as np


    # <actuator>
    #     <position name="shoulder_pos" joint="joint_shoulder" kp="100" />
    #     <position name="bicep_pos" joint="joint_bicep" kp="100" />
    #     <position name="forearm_pos" joint="joint_forearm" kp="100" />
    #     <position name="gripper_pos" tendon="gripper_tendon" kp="100" />
    # </actuator>


def calc_link_lengths():

    model = mujoco.MjModel.from_xml_path("main.xml")
    data  = mujoco.MjData(model)

    # Set all joints to zero (home pose)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # The joint bodies in order (proximal → distal)
    joint_bodies = [
        "joint_shoulder",
        "joint_bicep",
        "joint_forearm",
        "joint_gripper1",
    ]

    link_lengths = {
        "L0": 0.0,  # base height (world → joint_hip)
        "L1": 0.0,
        "L2": 0.0,
        "L3": 0.0,
        "L4": 0.0,
    }

    print("=== Joint World Positions (at zero pose) ===\n")

    # Get world position of each joint
    joint_positions = {}
    for jnt_name in joint_bodies:
        jnt_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        body_id = model.jnt_bodyid[jnt_id]
        pos     = data.xpos[body_id].copy()
        joint_positions[jnt_name] = pos
        print(f"  {jnt_name:20s}  →  {pos}")

    # Also get gripper (end-effector) body position
    gripper_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
    gripper_pos = data.xpos[gripper_id].copy()
    joint_positions["end_effector"] = gripper_pos
    print(f"  {'gripper (EE)':20s}  →  {gripper_pos}")

    # Compute link lengths as distance between consecutive joint positions
    print("\n=== Link Lengths (Euclidean distance between joints) ===\n")
    keys = list(joint_positions.keys())
    for i in range(len(keys) - 1):
        p1 = joint_positions[keys[i]]
        p2 = joint_positions[keys[i+1]]
        length = np.linalg.norm(p2 - p1)
        link_lengths[f"L{i+1}"] = length
        print(f"  L{i+1}  {keys[i]:20s} → {keys[i+1]:20s}  =  {length:.4f} m  ({length*100:.2f} cm)")
    print("---=-=--=\n")
    print(link_lengths)

    print("\n=== Summary for IK ===")
    print(f"  Base height (world → joint_shoulder) : {joint_positions['joint_shoulder'][2]:.4f} m")
    link_lengths["L0"] = joint_positions["joint_shoulder"][2]  # base height
    for i in range(len(keys) - 1):
        p1 = joint_positions[keys[i]]
        p2 = joint_positions[keys[i+1]]
        print(f"  L{i+1} = {np.linalg.norm(p2-p1):.4f} m")

    return link_lengths