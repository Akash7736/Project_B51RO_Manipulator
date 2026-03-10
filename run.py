import random

import mujoco
import mujoco.viewer
import numpy as np
import time
import dim
import ik


model = mujoco.MjModel.from_xml_path("main.xml")
data = mujoco.MjData(model)

# Print basic model info
print(f"Model name: {model.opt.timestep}")
print(f"Number of joints: {model.njnt}")
print(f"Number of actuators: {model.nu}")
print(f"Number of bodies: {model.nbody}")

def get_body_pos(data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def check_fk():

    theta_1_range = model.jnt_range[0]  # shoulder
    theta_2_range = model.jnt_range[1]  # bicep
    theta_3_range = model.jnt_range[2]  # forearm
    # theta_1 = random.uniform(*theta_1_range)
    # theta_2 = random.uniform(*theta_2_range)
    # theta_3 = random.uniform(*theta_3_range)
    theta_1 = np.pi
    theta_2 = np.pi/2
    theta_3 = -np.pi/2


    # theta_1 = 0
    # theta_2 = 0
    # theta_3 = 0


    print(f"Testing FK with random joint angles:")
    print(f"  theta_1 (shoulder): {theta_1:.3f} rad")
    print(f"  theta_2 (bicep):   {theta_2:.3f} rad")
    print(f"  theta_3 (forearm): {theta_3:.3f} rad")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        
        # Let robot settle for ~2 seconds before comparing
        settled = False
        while viewer.is_running() and time.time() - start < 180:
            data.ctrl[:] = [theta_1, theta_2, theta_3, 0]
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            # Only compare after settling
            if not settled and time.time() - start > 2.0:
                settled = True

            if settled:
                end_effector_pos = get_body_pos(data, "end_effector")
                thetas = [theta_1, theta_2, theta_3]
                fk_ee_pos = ik.fk(thetas)
                err = np.linalg.norm(end_effector_pos - fk_ee_pos)
                print(f"Sim: {np.round(end_effector_pos, 4)}  |  FK: {np.round(fk_ee_pos, 4)}  |  err: {err:.5f}")


def run_passive():
        # Run this to get the quat of gripper when it WAS correct (comment out compiler line temporarily)

    g1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper1_Gripper1")
    print("gripper1 xquat:", data.xquat[g1_id])  # w, x, y, z
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and 'gripper' in name.lower():
            print(f"\n{name}:")
            print(f"  pos:  {data.xpos[i]}")
            print(f"  xmat: {data.xmat[i].reshape(3,3)}")
    with mujoco.viewer.launch_passive(model, data) as viewer:  # 👈 launch_passive, not launch
        print("\nRunning simulation with control panel...")
        start = time.time()
        while viewer.is_running() and time.time() - start < 180:
            # data.ctrl[:] = [0 ,0 , 0, 0]
            mujoco.mj_step(model, data)

            # print("ctrl (target) :", data.ctrl[0])       # what you set
            # print("qpos (actual) :", data.qpos[0])        # where it actually is
            # print("error (rad)   :", data.ctrl[0] - data.qpos[0])
            viewer.sync()
            time.sleep(model.opt.timestep)  # sync sleep to timestep

if __name__ == "__main__":
    # Reset simulation to initial state
    print("initial data.ctrl : ", data.ctrl)
    mujoco.mj_resetData(model, data)

    # Calculate link lengths
    link_lengths = dim.calc_link_lengths()


    # Test control without viewer
    print("Initial qpos:", data.qpos[:4])
    data.ctrl[:] = [0, np.pi/2, -np.pi/2, 0]
    print("Set ctrl to:", data.ctrl)
    for i in range(1000):
        mujoco.mj_step(model, data)
    print("qpos after 1000 steps:", data.qpos[:4])

    # Reset before running passive
    # mujoco.mj_resetData(model, data)


    # Reset to zero, step once, print ALL body positions
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    print("qpos0 (default offsets):", model.qpos0)
    print("jnt_range:", model.jnt_range)

    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"{i} {name}: range={model.jnt_range[i]}")

    data.ctrl[:] = [0, 0, 0, 0]
    for i in range(2000):
        mujoco.mj_step(model, data)

    print("=== All body xpos (world frame) at zero angles ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  {i} {name}: {data.xpos[i]}")

    print("\n=== qpos (actual joint angles after settling) ===")
    print(data.qpos[:4])

    print("\n=== FK at zero angles ===")
    print(ik.fk([0, 0, 0]))
    # run_passive()         # just let it fall under gravity
    check_fk()            # test FK with random joint angles
