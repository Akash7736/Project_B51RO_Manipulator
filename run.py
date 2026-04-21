import random

import mujoco
import mujoco.viewer
import numpy as np
import time
import dim
import ik
from detect import *

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



# def localize_block(image_bgr, R_cam, t_cam, fovy=120, img_w=640, img_h=480):
#     # 1. Detect
#     result = detect_block_2d(image_bgr)
#     if result is None:
#         return None
#     cx, cy, w, h = result

#     # 2. Depth from known size
#     depth = estimate_depth_from_size(h, real_height=0.09, fovy_deg=fovy, img_height=img_h)

#     # 3. 2D → 3D cam frame
#     p_cam = pixel_to_cam_frame(cx, cy, depth, fovy, img_w, img_h)

#     # 4. Cam → world
#     p_world = cam_to_world(p_cam, R_cam, t_cam)

#     return p_world

def localize_block(image_bgr, R_cam, t_cam, fovy=90, img_w=640, img_h=480):
    result = detect_block_2d(image_bgr)
    if result is None:
        return None
    cx, cy, w, h = result

    # Use bottom center — where block meets ground
    cx_base = cx
    cy_base = cy + h // 2

    # Compute depth from pixel row using camera height
    depth = pixel_row_to_depth(cy_base, t_cam, R_cam, fovy_deg=fovy, img_h=img_h)
    if depth is None:
        print("Pixel above horizon, can't compute depth")
        return None
    
    print(f"Estimated depth from row geometry: {depth:.4f}m")

    # Back-project to 3D
    p_cam = pixel_to_cam_frame(cx_base, cy_base, depth, fovy, img_w, img_h)
    p_world = cam_to_world(p_cam, R_cam, t_cam)

    # Override Z with known block center height
    p_world[2] = 0.09   # block half-height

    return p_world



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

def is_reachable(target_pos, tolerance=0.03):
    pts = []
    for t1 in np.linspace(-np.pi, np.pi, 20):
        for t2 in np.linspace(-np.pi, np.pi, 20):
            for t3 in np.linspace(-np.pi, np.pi, 20):
                pts.append(ik.fk([t1, t2, t3]))
    pts = np.array(pts)
    dist = np.linalg.norm(pts - np.array(target_pos), axis=1).min()
    return dist < tolerance, dist



# def run_with_control():
#     JOINT_LIMITS = [
#         (-np.pi,     np.pi),
#         (-np.pi/2,   np.pi/2),
#         (-np.pi/2,   np.pi/2),
#     ]

#     target_pos = [0.25, 0.25, 0.1]
#     # target_pos = [0.3, 0.1, 0.5]  
#         # Then before calling IK:
#     reachable, dist = is_reachable(target_pos)
#     if not reachable:
#         print(f"Target unreachable — closest point is {dist:.4f}m away")
#         return
    
      
#     solved3, hist3, ok3 = ik.inverse_kinematics(
#         target_pos=target_pos,
#         theta_init=None,
#         max_iter=15000,
#         alpha=0.005,
#         tol=1e-5,
#         joint_limits=JOINT_LIMITS,
#     )
#     print(f"\nIK solved: {ok3} in {len(hist3)} iterations")
#     target_positions = np.array(solved3)
#     achieved3 = ik.fk(solved3)
#     print("\n=== IK Solution for Target Position", target_pos, "===")
#     for i, angle in enumerate(target_positions):
#         print(f"  Joint {i+1}: {angle:.4f} rad  ({np.degrees(angle):.2f}°)")
#     print(f"  Achieved position: {achieved3}")

#     # ── Trajectory Setup ──────────────────────────────────────────
#     dt          = model.opt.timestep
#     print(f"\nSimulation timestep: {dt:.4f} seconds")
#     T_move      = 10.0   # seconds to complete the motion — tune this
    
#     # Start from wherever the joints currently are
#     q_start = data.qpos[:3].copy()
#     q_end   = target_positions

#     positions, velocities, _ = ik.quintic_trajectory(q_start, q_end, T_move, dt)
    
#     print(f"\nTrajectory: {len(positions)} steps over {T_move}s")
#     # ─────────────────────────────────────────────────────────────

#     print_flag  = True
#     traj_idx    = 0
#     sim_start   = time.time()

#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         print("\nRunning with trajectory control...")
#         start = time.time()

#         while viewer.is_running() and time.time() - start < 180:
            
#             # ── Feed trajectory setpoint ──────────────────────────
#             if traj_idx < len(positions):
#                 # print("pOSITIONS")
#                 data.ctrl[:3] = positions[traj_idx]
#                 traj_idx += 1
#             else:
#                 # Hold final position once trajectory is complete
#                 data.ctrl[:3] = q_end
#                 data.ctrl[3] = 0.5  # open gripper
#             # ─────────────────────────────────────────────────────

#             mujoco.mj_step(model, data)
#             curr_time = time.time() - sim_start

#             # if curr_time > T_move + 1.0 and print_flag:
#             gripper_pos = get_body_pos(data, "end_effector")
#             print("End effector position:", gripper_pos)
#             print(f"target position: {target_pos}  |  error: {np.linalg.norm(gripper_pos - target_pos):.5f}")
            
#                 # print_flag = False

#             viewer.sync()
#             time.sleep(dt)


def run_with_control():
    JOINT_LIMITS = [
        (-np.pi,     np.pi),
        (-np.pi/2,   np.pi/2),
        (-np.pi/2,   np.pi/2),
    ]

    # ── Setup renderer ────────────────────────────────────────────
    renderer = mujoco.Renderer(model, height=480, width=640)

    # ── Reset to initial state for clean ArUco image ─────────────
    # Block initial pos in XML (z=0.1) is already near floor.
    # At t=0 the arm is upright and clear of both markers.
    mujoco.mj_resetData(model, data)

    # ── Render camera frame ───────────────────────────────────────
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "base_cam_3")
    R_cam_nominal = data.cam_xmat[cam_id].reshape(3, 3).copy()
    t_cam_nominal = data.cam_xpos[cam_id].copy()
    renderer.update_scene(data, camera="base_cam_3")
    image_rgb = renderer.render()
    image_bgr = image_rgb[:, :, ::-1]  # RGB → BGR for OpenCV
    cv2.imwrite("rendered_view.png", image_bgr)  # save for debugging

    # ── Localize block using ArUco markers as reference ──────────
    target_pos_world, target_pos_base = localize_block_aruco(
        image_bgr, R_cam_nominal, t_cam_nominal, fovy=120, img_w=640, img_h=480
    )
    if target_pos_world is None:
        print("Block not detected via ArUco pipeline! Check debug_aruco.png / debug_detection.png")
        return

    target_pos = target_pos_world  # IK operates in world frame

    # ── Validation: compare against ground truth ──────────────────
    gt_pos = get_body_pos(data, "box_block")
    print(f"[GT]  Block world pos (sim)  : {np.round(gt_pos,          4)}")
    print(f"[EST] Block world pos        : {np.round(target_pos,       4)}")
    print(f"[EST] Block wrt robot base   : {np.round(target_pos_base,  4)}")
    print(f"[ERR] Localization error     : {np.linalg.norm(target_pos - gt_pos):.4f} m")

    # ── rest is same as before ────────────────────────────────────
    reachable, dist = is_reachable(target_pos)
    if not reachable:
        print(f"Target unreachable — closest point is {dist:.4f}m away")
        return

    solved3, hist3, ok3 = ik.inverse_kinematics(
        target_pos=target_pos,
        theta_init=None,
        max_iter=15000,
        alpha=0.005,
        tol=1e-5,
        joint_limits=JOINT_LIMITS,
    )
    print(f"\nIK solved: {ok3} in {len(hist3)} iterations")
    target_positions = np.array(solved3)

    dt     = model.opt.timestep
    T_move = 10.0
    q_start = data.qpos[:3].copy()
    q_end   = target_positions
    positions, velocities, _ = ik.quintic_trajectory(q_start, q_end, T_move, dt)

    traj_idx = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nRunning with trajectory control...")
        start = time.time()

        while viewer.is_running() and time.time() - start < 180:
            if traj_idx < len(positions):
                data.ctrl[:3] = positions[traj_idx]
                traj_idx += 1
            else:
                data.ctrl[:3] = q_end
                data.ctrl[3] = 0.5

            mujoco.mj_step(model, data)
            gripper_pos = get_body_pos(data, "end_effector")
            # print(f"EE: {np.round(gripper_pos,4)}  target: {np.round(target_pos,4)}  err: {np.linalg.norm(gripper_pos - target_pos):.4f}")
            viewer.sync()
            time.sleep(dt)


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

    data.ctrl[:] = [0, 0, 0, -0.5]
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
    # check_fk()            # test FK with random joint angles
    run_with_control()     # test IK + trajectory control to target position
