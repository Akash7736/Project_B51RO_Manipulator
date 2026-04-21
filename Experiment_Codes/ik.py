import numpy as np
# from dim import calc_link_lengths

# # LINK_LENGTHS = calc_link_lengths()
# L0 = LINK_LENGTHS["L0"]  # base height (world → joint_shoulder)
# L1 = LINK_LENGTHS["L1"]  # joint_shoulder → joint_bicep
# L2 = LINK_LENGTHS["L2"]  # joint_bicep → joint_forearm
# L3 = LINK_LENGTHS["L3"]  # joint_forearm → joint_gripper1
# L4 = LINK_LENGTHS["L4"]  # joint_gripper1 → end_effector


L0 = 0.05
L1 = 0.06
L2 = 0.08
L3 = 0.05
L4 = 0.05

#    joint_bodies = [
#         "joint_shoulder",
#         "joint_bicep",
#         "joint_forearm",
#         "joint_gripper1",
#     ]

def Rx(angle):
    return np.array([[1,0,0], [0,np.cos(angle),-np.sin(angle)], [0,np.sin(angle),np.cos(angle)]])
def Ry(angle):
    return np.array([[np.cos(angle),0,np.sin(angle)], [0,1,0], [-np.sin(angle),0,np.cos(angle)]])
def Rz(angle):
    return np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])

def Tx(distance):
    return np.array([[1,0,0,distance], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
def Ty(distance):
    return np.array([[1,0,0,0], [0,1,0,distance], [0,0,1,0], [0,0,0,1]])
def Tz(distance):
    return np.array([[1,0,0,0], [0,1,0,0], [0,0,1,distance], [0,0,0,1]])

def Homo_Matrix(R,d):
    return np.vstack((np.hstack((R, d[:, None])), [0, 0, 0, 1]))


# def Transformation_Matrix(theta_1, theta_2, theta_3): # Twist, Revolute, Revolute, Revolute-grip

#     # theta_1 = 0.0
#     R01 =np.eye(3)  @ Rz(theta_1) 
#     d01 = np.array([0, 0, L0])
#     H01 = Homo_Matrix(R01, d01)

#     T01 = H01
#     # print("T01:\n", T01)

#     # theta_2 =  np.pi/2
#     R12 =  np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) @ Rz(theta_2) 
#     d12 = np.array([0, 0, L1])
#     H12 = Homo_Matrix(R12, d12)
#     T12 = H12
#     # print("T02:\n", T01@T12) 

#     # theta_3 = -np.pi/2 
#     R23 = np.eye(3) @ Rz(theta_3)
#     d23 = np.array([L2, 0, 0])
#     H23 = Homo_Matrix(R23, d23)
#     T23 = H23
#     # print("T03:\n", T01@T12@T23)

#     # theta_4 = 0.0
#     R34 =  np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) @ Rz(theta_3) 
#     d34 = np.array([L3+L4, 0, 0])
#     H34 = Homo_Matrix(R34, d34)
#     T34 = H34
#     # print("T04:\n", T01@T12@T23@T34)



#     # print("T06:\n", T01@T12@T23@T34@T45@T56)

#     T = T01 @ T12 @ T23 @ T34 

#     return T 


def Transformation_Matrix(theta_1, theta_2, theta_3):

    # Joint 1: shoulder — rotates around Z, offset straight up
    R01 = Rz(theta_1)
    d01 = np.array([0.0, 0.0, L0])          # world → shoulder
    T01 = Homo_Matrix(R01, d01)

    # Joint 2: bicep — rotates around X, offset is (-0.05, 0, 0.1) in shoulder frame
    R12 = Rx(theta_2)
    d12 = np.array([-0.05, 0.0, L1])        # shoulder → bicep
    T12 = Homo_Matrix(R12, d12)

    # Joint 3: forearm — rotates around X, offset is (0.025, 0, 0.15) in bicep frame
    R23 = Rx(theta_3)
    d23 = np.array([0.025, 0.0, L2])        # bicep → forearm
    T23 = Homo_Matrix(R23, d23)

    # Fixed: forearm → end_effector (0.02, 0, 0.190) in forearm frame
    R3e = np.eye(3)
    d3e = np.array([0.02, 0.0, L3 + L4])   # forearm → end effector
    T3e = Homo_Matrix(R3e, d3e)

    return T01 @ T12 @ T23 @ T3e

def fk(thetas):
    """Forward kinematics: returns end-effector (x, y, z) position."""
    T = Transformation_Matrix(*thetas)
    return T[:3, 3]


# ─── Jacobian (numerical, position-only, 3×5) ─────────────────────────────────

def compute_jacobian(thetas, delta=1e-5):
    """
    Numerical Jacobian via central differences.
    
    Returns J (3×5) where J[:,i] = dp/d(theta_i).
    """
    thetas = np.array(thetas, dtype=float)
    n = len(thetas)
    p0 = fk(thetas)
    J = np.zeros((3, n))
    for i in range(n):
        t_plus  = thetas.copy(); t_plus[i]  += delta
        t_minus = thetas.copy(); t_minus[i] -= delta
        J[:, i] = (fk(t_plus) - fk(t_minus)) / (2 * delta)
    return J


# ─── Inverse Kinematics (damped least-squares / Levenberg–Marquardt) ──────────

def inverse_kinematics(
    target_pos,
    theta_init=None,
    max_iter=2000,
    tol=1e-8,
    alpha=0.5,              # base step size (learning rate)
    lambda_damp=0.01,       # damping factor for DLS
    lambda_max=1.0,         # max damping (adaptive ceiling)
    max_step=0.2,           # max joint delta per iteration (radians)
    joint_limits=None,      # list of (min, max) per joint, or None
):
    """
    Damped Least-Squares (DLS) IK for position-only (3 DOF target, 5 joints).

    Uses adaptive damping: λ increases when error grows (diverging) and
    decreases when converging, balancing speed vs. stability.
    Step clamping prevents large jumps near singularities.

    Parameters
    ----------
    target_pos   : array-like (3,)  – desired end-effector position [x, y, z]
    theta_init   : array-like (5,)  – initial joint angles (radians); zeros if None
    max_iter     : int              – maximum iterations
    tol          : float            – position error tolerance to stop
    alpha        : float            – step-size scaling
    lambda_damp  : float            – initial damping coefficient
    lambda_max   : float            – maximum allowed damping
    max_step     : float            – per-joint step clamp (radians)
    joint_limits : list of (lo, hi) – optional per-joint limits in radians

    Returns
    -------
    thetas       : np.ndarray (5,)  – solved joint angles
    history      : list of floats   – position error at each iteration
    success      : bool             – True if converged within tol
    """
    target = np.array(target_pos, dtype=float)

    if theta_init is None:
        thetas = np.zeros(3)
    else:
        thetas = np.array(theta_init, dtype=float)

    history = []
    prev_err = np.inf
    lam = lambda_damp

    for iteration in range(max_iter):
        pos = fk(thetas)
        error = target - pos
        err_norm = np.linalg.norm(error)
        history.append(err_norm)

        if err_norm < tol:
            return thetas, history, True

        # Adaptive damping: increase λ if diverging, decrease if converging
        if err_norm > prev_err:
            lam = min(lam * 2.0, lambda_max)
        else:
            lam = max(lam * 0.9, lambda_damp)
        prev_err = err_norm

        J = compute_jacobian(thetas)

        # Damped least-squares pseudo-inverse:  Jᵀ(JJᵀ + λ²I)⁻¹
        JJT = J @ J.T
        dls_inv = J.T @ np.linalg.inv(JJT + lam**2 * np.eye(3))

        delta_theta = alpha * dls_inv @ error

        # Clamp per-joint step to avoid large jumps near singularities
        delta_theta = np.clip(delta_theta, -max_step, max_step)

        thetas = thetas + delta_theta

        # Wrap angles back into [-π, π] using circular arithmetic.
        # This is correct for periodic joints: arctan2(sin,cos) is lossless
        # and avoids drift beyond ±π that breaks FK trigonometry.
        if joint_limits is None:
            thetas = np.arctan2(np.sin(thetas), np.cos(thetas))
        else:
            # With explicit limits, clamp instead of wrap
            for i, (lo, hi) in enumerate(joint_limits):
                thetas[i] = np.clip(thetas[i], lo, hi)

    # Return best result even if not converged
    return thetas, history, False


def workspace_sample(n=500, seed=42):
    """
    Sample reachable positions by running FK on random joint configs.
    Useful for visualising or verifying reachability of a target.

    Returns array of shape (n, 3).
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=(n, 5))
    return np.array([fk(a) for a in angles])



def quintic_trajectory(q_start, q_end, T, dt):
    """
    Generates a quintic (5th order) polynomial trajectory between two joint configurations.
    Guarantees: position, velocity, AND acceleration continuity (zero at start and end).
    
    Args:
        q_start : np.array — starting joint angles (radians)
        q_end   : np.array — target joint angles (radians)
        T       : float    — total movement duration (seconds)
        dt      : float    — timestep (seconds), use model.opt.timestep
    
    Returns:
        positions     : (N, DOF) array of joint angles over time
        velocities    : (N, DOF) array of joint velocities over time
        accelerations : (N, DOF) array of joint accelerations over time
    """
    q_start = np.array(q_start)
    q_end   = np.array(q_end)
    
    time_steps = np.arange(0, T + dt, dt)
    N   = len(time_steps)
    DOF = len(q_start)
    
    positions     = np.zeros((N, DOF))
    velocities    = np.zeros((N, DOF))
    accelerations = np.zeros((N, DOF))
    
    for i, t in enumerate(time_steps):
        # Normalized time [0, 1]
        tau = t / T
        tau = np.clip(tau, 0.0, 1.0)
        
        # Quintic basis polynomials
        # These satisfy: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
        s     =  10*tau**3 - 15*tau**4 + 6*tau**5
        s_dot = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
        s_ddot= (60*tau    - 180*tau**2 + 120*tau**3) / T**2
        
        positions[i]     = q_start + (q_end - q_start) * s
        velocities[i]    = (q_end - q_start) * s_dot
        accelerations[i] = (q_end - q_start) * s_ddot
    
    return positions, velocities, accelerations


# ─── Test: IK with FK validation ──────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=" * 60)
#     print("  Kinematics Test Suite")
#     print("=" * 60)

#     # ── Test 1: FK at a known configuration ──────────────────────
#     print("\n[Test 1] FK at home configuration")
#     home = [0.0, -np.pi/2, np.pi/2, 0.0, 0.0]
#     T_home = Transformation_Matrix(*home)
#     print(f"  Angles (deg): {[round(np.degrees(a), 1) for a in home]}")
#     print(f"  End-effector position (FK): {T_home[:3, 3]}")
#     print(f"  Transform:\n{np.round(T_home, 4)}")

#     # ── Test 2: IK → FK round-trip ───────────────────────────────
#     print("\n[Test 2] IK → FK round-trip validation")

#     # Pick a reachable target by running FK on a known config
#     known_thetas = np.array([0.3, -1.0, 1.2, 0.2, -0.4])
#     target = fk(known_thetas)
#     print(f"  Known thetas (deg): {[round(np.degrees(a), 1) for a in known_thetas]}")
#     print(f"  Target position (from FK): {np.round(target, 5)}")

#     # Solve IK from a perturbed initial guess (seeded for reproducibility)
#     rng = np.random.default_rng(0)
#     theta_init = known_thetas + rng.uniform(-0.3, 0.3, size=5)
#     solved_thetas, history, success = inverse_kinematics(
#         target_pos=target,
#         theta_init=theta_init,
#         max_iter=2000,
#         tol=1e-5,
#     )

#     # Validate with FK
#     achieved_pos = fk(solved_thetas)
#     position_error = np.linalg.norm(target - achieved_pos)

#     print(f"\n  IK converged: {success}  (iterations: {len(history)})")
#     print(f"  Solved thetas (deg): {[round(np.degrees(a), 1) for a in solved_thetas]}")
#     print(f"  Achieved position (FK validation): {np.round(achieved_pos, 5)}")
#     print(f"  Position error: {position_error:.2e}")
#     print(f"  PASS ✓" if position_error < 1e-3 else f"  FAIL ✗  (error too large)")

#     # ── Test 3: Jacobian sanity check ────────────────────────────
#     print("\n[Test 3] Jacobian shape and finite-difference sanity")
#     J = compute_jacobian(known_thetas)
#     print(f"  Jacobian shape: {J.shape}  (expected 3×5)")
#     print(f"  Jacobian:\n{np.round(J, 4)}")
#     rank = np.linalg.matrix_rank(J)
#     print(f"  Jacobian rank: {rank}  (full = 3)")

#     # ── Test 4: IK from zero-init, target chosen from real workspace ─
#     print("\n[Test 4] IK from zero-init to workspace-verified target")

#     # Use FK on a config comfortably reachable from zero-init
#     reachable_thetas = np.array([0.0, -0.5, 0.8, 0.3, 0.2])
#     arb_target = fk(reachable_thetas)
#     print(f"  Target (from FK of [{ ', '.join(f'{np.degrees(a):.1f}deg' for a in reachable_thetas) }]):")
#     print(f"    {np.round(arb_target, 5)}")

#     solved2, hist2, ok2 = inverse_kinematics(
#         target_pos=arb_target,
#         theta_init=None,    # cold start from zeros
#         max_iter=3000,
#         tol=1e-5,
#     )
#     achieved2 = fk(solved2)
#     err2 = np.linalg.norm(arb_target - achieved2)
#     print(f"  Solved thetas (deg): {[round(np.degrees(a), 1) for a in solved2]}")
#     print(f"  Achieved: {np.round(achieved2, 5)}")
#     print(f"  Error: {err2:.2e}  |  Converged: {ok2}  |  Iterations: {len(hist2)}")
#     print(f"  PASS ✓" if err2 < 1e-3 else f"  FAIL ✗  (error too large)")

#     # ── Test 5: workspace extent (informational) ──────────────────
#     print("\n[Test 5] Workspace extent (500 random configs)")
#     ws = workspace_sample(500)
#     print(f"  X range: [{ws[:,0].min():.4f}, {ws[:,0].max():.4f}] m")
#     print(f"  Y range: [{ws[:,1].min():.4f}, {ws[:,1].max():.4f}] m")
#     print(f"  Z range: [{ws[:,2].min():.4f}, {ws[:,2].max():.4f}] m")
#     max_reach = np.linalg.norm(ws, axis=1).max()
#     print(f"  Max reach (from origin): {max_reach:.4f} m")
#     print(f"  Theoretical max (L1+L2+L3+L4+L5): {L1+L2+L3+L4+L5:.4f} m")

#     print("\n" + "=" * 60)


#     JOINT_LIMITS = [
#         (-np.pi,     np.pi),     # J1 twist  — full rotation
#         (-np.pi/2,   np.pi/2),   # J2 shoulder — ±90°
#         (-np.pi/2,     np.pi/2),     # J3 elbow
#         (-np.pi,     np.pi),     # J4 forearm twist
#         (-np.pi/2,   np.pi/2),   # J5 wrist
#     ]


#     target_pos = [0.3, 0.1, 0.5]   # example target
#     solved3, hist3, ok3 = inverse_kinematics(
#     target_pos=target_pos,
#     theta_init=None,    # cold start from zeros
#     max_iter=9000,
#     alpha=0.01,
#     tol=1e-5,
#     joint_limits=JOINT_LIMITS,
# )
    
#     print(f"\n[Test 6] IK from zero-init to arbitrary target {target_pos}")
#     achieved3 = fk(solved3)
#     err3 = np.linalg.norm(np.array(target_pos) - achieved3)
#     print(f"  Solved thetas (deg): {[round(np.degrees(a), 1) for a in solved3]}")
#     print(f"  Achieved: {np.round(achieved3, 5)}")
#     print(f"  Error: {err3:.2e}  |  Converged: {ok3}  |  Iterations: {len(hist3)}")
#     print(f"  PASS ✓" if err3 < 1e-5 else f"  FAIL ✗  (error too large)") 