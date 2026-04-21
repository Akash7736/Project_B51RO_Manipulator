# Simulation

The simulation environment is built on **MuJoCo** (Multi-Joint Dynamics with Contact), a physics engine chosen for its accurate rigid-body dynamics, fast contact resolution, and tight integration with Python for programmatic control. The scene is defined in `main.xml` and a collection of modular include files under `models/`, each describing a link in the kinematic chain as a mesh-backed rigid body.

## Robot Model

The manipulator is a **3-DOF serial arm** with a parallel-jaw gripper, totalling four actuated degrees of freedom. The kinematic chain is:

| Joint | Axis | Range |
|---|---|---|
| Shoulder | Z (rotation) | ±180° |
| Bicep | X (elevation) | ±90° |
| Forearm | X (elevation) | ±90° |
| Gripper | coupled (tendon) | symmetric open/close |

Link geometry is sourced from STL meshes scaled 2× and loaded via the MuJoCo mesh asset system. The gripper fingers are coupled through a fixed tendon (`coef = ±1`) so that both fingers close symmetrically without requiring independent low-level commands.

All joints are driven by **position-control actuators** (`kp = 300` for arm joints, `kp = 25` for the gripper), which behave as high-gain PD servos. Joint damping of 1 N·m·s/rad is applied for numerical stability.

## Scene Layout

The scene includes a flat ground plane with a checkerboard material, a target block (60 × 60 × 100 mm box) placed at a known initial position, two **ArUco fiducial marker tiles** (30 × 30 cm) embedded flush with the floor on either side of the workspace, and an overhead RGB camera mounted on a fixed stand at (0, 0.40, 0.30) m, tilted −45° downward with a 120° field of view.

## Perception Pipeline

The camera renders 640 × 480 RGB frames directly from the MuJoCo offscreen renderer. Object localization proceeds in three stages:

1. **ArUco pose estimation** — Each marker tile is individually deskewed via a perspective warp into a 200 × 200 px upright patch with a 20 px white quiet zone, then decoded using OpenCV's `DICT_4X4_50` dictionary. Known 3D corner positions of both detected markers are passed to `solvePnP` to refine the camera extrinsics (rotation `R`, translation `t`). Frames with reprojection error above 15 px are discarded.

2. **Block 2D detection** — The block is segmented from the camera image by HSV colour filtering (hue 5–20°, saturation and value > 150) restricted to the lower half of the frame. The centroid of the largest valid contour gives pixel coordinates `(cx, cy)`.

3. **3D localization** — The pixel is back-projected through the pinhole camera model into a ray in the world frame, which is then intersected with the `Z = 0` ground plane. The block's height (0.09 m) is substituted for the vertical coordinate, yielding a 3D target position in the robot base frame.

## Inverse Kinematics

Joint angles are computed by a **Damped Least-Squares (DLS)** solver with adaptive damping. At each iteration the 3 × 3 numerical Jacobian is evaluated by central finite differences, and the joint update is

$$\Delta\theta = \alpha \, J^\top (J J^\top + \lambda^2 I)^{-1} \, e$$

with step size `α = 0.005` rad and initial damping `λ = 0.01`. If the position error grows between iterations the damping is doubled (up to `λ_max = 1.0`); otherwise it is decayed by a factor of 0.9, balancing convergence speed against singularity avoidance. Convergence is declared when `‖e‖ < 10⁻⁵` m.

## Trajectory Execution

The solved joint configuration is connected to the home pose via a **quintic (5th-order) polynomial** trajectory, which enforces zero velocity and acceleration at both endpoints. At a simulation timestep of 2 ms the trajectory is sampled into approximately 5,000 waypoints, which are issued as servo setpoints in real time through the MuJoCo viewer loop. Upon reaching the target the gripper is commanded to close, completing the pick operation.
