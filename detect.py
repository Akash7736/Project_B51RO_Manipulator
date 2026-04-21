
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ArUco marker world-frame corner positions (matches main.xml geom placement)
#
# Both markers lie flat on the floor (Z-up world frame).
# geom size="0.05 0.05 0.001" → 10 cm × 10 cm tile, top face at z=0.002.
# Corner order: 0=TL, 1=TR, 2=BR, 3=BL  (clockwise from top-left when
# viewed from above with camera looking toward −Y).
#   TL = (cx−s, cy+s)   TR = (cx+s, cy+s)
#   BL = (cx−s, cy−s)   BR = (cx+s, cy−s)
# Marker 0: center (−0.20, 0.05)   Marker 1: center (+0.20, 0.05)
# ---------------------------------------------------------------------------
_S = 0.15   # half-side of marker tile (metres) — 30 cm × 30 cm tile
_Z = 0.002  # top-face height

ARUCO_MARKER_WORLD_CORNERS = {
    # ID 0 — left side marker  (geom aruco_left,  file aruco_marker_0.png)
    0: np.array([
        [-0.20 - _S,  0.15 + _S, _Z],  # TL
        [-0.20 + _S,  0.15 + _S, _Z],  # TR
        [-0.20 + _S,  0.15 - _S, _Z],  # BR
        [-0.20 - _S,  0.15 - _S, _Z],  # BL
    ], dtype=np.float32),
    # ID 2 — right side marker  (geom aruco_right, file aruco_marker_1.png holds ID=2)
    2: np.array([
        [ 0.20 - _S,  0.15 + _S, _Z],  # TL
        [ 0.20 + _S,  0.15 + _S, _Z],  # TR
        [ 0.20 + _S,  0.15 - _S, _Z],  # BR
        [ 0.20 - _S,  0.15 - _S, _Z],  # BL
    ], dtype=np.float32),
}

# Robot base body origin in world frame (from main.xml: <body pos="0 0 0.1">)
ROBOT_BASE_WORLD_POS = np.array([0.0, 0.0, 0.1])

def pixel_row_to_depth(cy, t_cam, R_cam, fovy_deg=90, img_h=480):
    """
    Given a pixel row cy, compute the depth to that point
    using known camera height above ground.
    
    Uses the vertical ray geometry:
    - camera is at height t_cam[2] above ground
    - pixel cy maps to a ray angle from optical axis
    - depth = camera_height / sin(angle_below_horizon)
    """
    fy = (img_h / 2) / np.tan(np.radians(fovy_deg / 2))
    
    # Angle of this pixel below the optical axis (in cam frame)
    # positive = below center = looking down
    dy = cy - img_h / 2   # pixels below center
    angle_from_axis = np.arctan(dy / fy)  # radians
    
    # Camera tilt angle below horizon (how much cam is tilted down)
    # R_cam[2] is the camera's forward direction in world
    # The angle below horizontal = arcsin(-R_cam[2][2]) since world Z is up
    cam_forward_world = R_cam @ np.array([0, 0, -1])  # MuJoCo -Z is forward
    tilt_angle = np.arcsin(-cam_forward_world[2])      # angle below horizontal
    
    # Total angle below horizontal for this pixel
    total_angle = tilt_angle + angle_from_axis
    
    if total_angle <= 0:
        return None  # pixel is above horizon
    
    # Camera height above ground
    cam_height = t_cam[2]
    
    # Depth along the ray to ground plane
    depth = cam_height / np.sin(total_angle)
    
    return depth



def pixel_to_world_ground(cx, cy, R_cam, t_cam, fovy_deg=90, img_w=640, img_h=480, ground_z=0.0):
    """
    Back-project pixel onto the world ground plane (Z = ground_z).
    Works for any static camera with known extrinsics.
    """
    fy = (img_h / 2) / np.tan(np.radians(fovy_deg / 2))
    fx = fy
    cx0, cy0 = img_w / 2, img_h / 2

    # Ray direction in camera frame (MuJoCo: looks along -Z)
    ray_cam = np.array([
         (cx - cx0) / fx,
        -(cy - cy0) / fy,   # flip Y
        -1.0                 # looking along -Z
    ])

    # Ray direction in world frame
    ray_world = R_cam @ ray_cam          # direction vector
    ray_origin = t_cam                   # camera position in world

    # Intersect with plane: world_Z = ground_z
    # ray_origin + t * ray_world → z component = ground_z
    # t = (ground_z - ray_origin[2]) / ray_world[2]
    
    if abs(ray_world[2]) < 1e-6:
        print("Ray is parallel to ground plane!")
        return None

    t = (ground_z - ray_origin[2]) / ray_world[2]

    if t < 0:
        print("Intersection is behind the camera!")
        return None

    p_world = ray_origin + t * ray_world
    return p_world


def detect_block_2d(image_bgr, debug=True):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    lower = np.array([5,  150, 150])
    upper = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Only look in bottom half
    h_img = mask.shape[0]
    mask[:h_img//2, :] = 0

    print("Mask nonzero pixels:", np.count_nonzero(mask))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        vis = image_bgr.copy()

    if not contours:
        if debug:
            cv2.imwrite("debug_detection.png", vis)
        return None

    # Draw ALL contours in yellow so you can see what's being picked up
    if debug:
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 1)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    print(f"Largest contour area: {area:.1f} px")

    if area < 50:
        if debug:
            cv2.imwrite("debug_detection.png", vis)
        return None

    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2

    if debug:
        # Bounding box in green
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Center dot in red
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        # Label
        cv2.putText(vis, f"block ({cx},{cy}) a={area:.0f}", 
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)
        # Midline showing top/bottom half split
        cv2.line(vis, (0, h_img//2), (vis.shape[1], h_img//2), (255, 0, 255), 1)
        # Mask overlay in blue tint
        mask_overlay = vis.copy()
        mask_overlay[mask > 0] = [255, 100, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        cv2.imwrite("debug_detection.png", vis)
        print("Saved debug_detection.png")

    return cx, cy, w, h

def estimate_depth_from_size(pixel_height, real_height=0.09, focal_length_px=None, fovy_deg=120, img_height=480):
    if focal_length_px is None:
        # derive from fovy
        focal_length_px = (img_height / 2) / np.tan(np.radians(fovy_deg / 2))
    
    depth = (real_height * focal_length_px) / pixel_height
    return depth


def pixel_to_cam_frame(cx, cy, depth, fovy_deg=120, img_w=640, img_h=480):
    # Camera intrinsics from fovy
    fy = (img_h / 2) / np.tan(np.radians(fovy_deg / 2))
    fx = fy  # square pixels assumed
    cx_principal = img_w / 2
    cy_principal = img_h / 2

    # Back-project
    x_cam = (cx - cx_principal) * depth / fx
    y_cam = (cy - cy_principal) * depth / fy
    z_cam = depth  # forward in cam frame for pinhole

    return np.array([x_cam, y_cam, z_cam])


def cam_to_world(p_cam, R_cam, t_cam):
    # R_cam, t_cam = extrinsics (cam pose in world)
    # MuJoCo: data.cam_xmat, data.cam_xpos
    p_world = R_cam @ p_cam + t_cam
    return p_world


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def _build_camera_matrix(fovy_deg=120, img_w=640, img_h=480):
    fy = (img_h / 2) / np.tan(np.radians(fovy_deg / 2))
    return np.array([[fy, 0, img_w / 2],
                     [0, fy, img_h / 2],
                     [0,  0,          1]], dtype=np.float64)


def _project_world_to_img(p_world, R_cam, t_cam, K):
    """
    Project a 3-D world point onto the image plane.
    MuJoCo convention: camera looks along -Z_cam; +X_cam = image right.
    Returns (u, v) pixel or None if behind the camera.
    """
    p_c = R_cam.T @ (np.asarray(p_world, float) - t_cam)
    depth = -p_c[2]            # MuJoCo -Z is forward → depth > 0 for visible pts
    if depth < 1e-6:
        return None
    u = K[0, 0] * p_c[0] / depth + K[0, 2]
    v = K[1, 2] - K[1, 1] * p_c[1] / depth   # Y flipped: image down ↔ cam -Y
    return np.array([u, v], dtype=np.float32)


# ---------------------------------------------------------------------------
# ArUco-based camera pose estimation via bird's-eye view (BEV) warp
# ---------------------------------------------------------------------------
# The oblique 45° angle of base_cam_3 makes direct ArUco detection unreliable.
# Strategy:
#   1. Warp the camera image to a top-down floor view using the nominal camera
#      pose (from MuJoCo).  ArUco detects reliably in this flat view.
#   2. Un-warp the detected BEV corners back to the original image.
#   3. solvePnP( original-image corners, known 3-D world corners ) → camera pose.
# ---------------------------------------------------------------------------

# BEV covers this world-floor area (extra margin keeps markers fully in frame):
_BEV_X = (-0.60, 0.60)   # metres, world X
_BEV_Y = (-0.10, 0.45)   # metres, world Y  (negative extends below robot base)
_BEV_PPM = 800            # pixels per metre in the BEV image


def _bev_size():
    w = int((_BEV_X[1] - _BEV_X[0]) * _BEV_PPM)
    h = int((_BEV_Y[1] - _BEV_Y[0]) * _BEV_PPM)
    return w, h                       # (770, 294) at default settings


def _compute_bev_homography(R_cam, t_cam, fovy_deg, img_w, img_h):
    """
    Compute H (image → BEV) and H_inv (BEV → image).

    BEV layout (all Z = 0):
      pixel (0,0)       ← world (_BEV_X[0], _BEV_Y[1])   top-left  = near-left
      pixel (W,0)       ← world (_BEV_X[1], _BEV_Y[1])   top-right = near-right
      pixel (W,H)       ← world (_BEV_X[1], _BEV_Y[0])   bot-right = far-right
      pixel (0,H)       ← world (_BEV_X[0], _BEV_Y[0])   bot-left  = far-left
    """
    K = _build_camera_matrix(fovy_deg, img_w, img_h)
    bev_w, bev_h = _bev_size()

    # Four floor corners that define the BEV rectangle
    world_pts = [
        np.array([_BEV_X[0], _BEV_Y[1], 0.0]),  # top-left  (near-left)
        np.array([_BEV_X[1], _BEV_Y[1], 0.0]),  # top-right (near-right)
        np.array([_BEV_X[1], _BEV_Y[0], 0.0]),  # bot-right (far-right)
        np.array([_BEV_X[0], _BEV_Y[0], 0.0]),  # bot-left  (far-left)
    ]
    bev_pts = np.float32([[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]])

    img_pts = []
    for p_w in world_pts:
        px = _project_world_to_img(p_w, R_cam, t_cam, K)
        if px is None:
            raise ValueError(f"BEV corner {p_w} behind camera — check camera pose.")
        img_pts.append(px)
    img_pts = np.float32(img_pts)

    H, _ = cv2.findHomography(img_pts, bev_pts)
    return H, np.linalg.inv(H), bev_w, bev_h


def _aruco_detect(image_bgr):
    """Detect DICT_4X4_50 markers. Returns (corners, ids)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(image_bgr)
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, dictionary, parameters=params)
    return corners, ids


# Pixel size of the deskewed per-marker warp before padding
_WARP_SIZE = 200
# White border added around the deskewed marker so ArUco has a clean quiet-zone
_WARP_PAD  = 20


def estimate_camera_pose_aruco(image_bgr, R_cam_nominal, t_cam_nominal,
                                fovy_deg=120, img_w=640, img_h=480, debug=True):
    """
    Estimate camera pose from ArUco markers via per-marker deskew + white-pad.

    Strategy
    --------
    For each known marker:
      1. Project world corners → image using nominal camera pose.
      2. Warp that region to a flat WARP_SIZE × WARP_SIZE image.
      3. Add a white border so the ArUco decoder has a clean quiet-zone.
      4. Detect ArUco.  Map detected corners back to original image space
         via the inverse homography (no circular dependency on nominal pose).
    Collect all valid 3D↔2D correspondences, then run solvePnP.

    Parameters
    ----------
    R_cam_nominal, t_cam_nominal : nominal camera extrinsics from MuJoCo.
        Used ONLY for the initial warp; the returned pose is independently
        computed from the detected corner positions.

    Returns
    -------
    R_cam, t_cam : refined camera pose in world frame  (None, None on failure)
    """
    K    = _build_camera_matrix(fovy_deg, img_w, img_h)
    dist = np.zeros(5, dtype=np.float64)
    obj_pts_list, img_pts_list, detected_ids = [], [], []
    WS, PAD = _WARP_SIZE, _WARP_PAD

    for mid, world_corners in ARUCO_MARKER_WORLD_CORNERS.items():
        # ── project nominal world corners to image ────────────────────────
        src_px = []
        for p_w in world_corners:
            px = _project_world_to_img(p_w, R_cam_nominal, t_cam_nominal, K)
            if px is None:
                break
            src_px.append(px)
        if len(src_px) < 4:
            continue   # marker behind camera

        dst_sq = np.float32([[0, 0], [WS, 0], [WS, WS], [0, WS]])
        M = cv2.getPerspectiveTransform(np.float32(src_px), dst_sq)

        # ── deskew + pad ──────────────────────────────────────────────────
        warped = cv2.warpPerspective(image_bgr, M, (WS, WS))
        padded = cv2.copyMakeBorder(warped, PAD, PAD, PAD, PAD,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # ── detect ArUco in padded deskewed image ─────────────────────────
        if debug:
            cv2.imwrite(f"debug_aruco_warp_id{mid}.png", padded)
        det_corners, det_ids = _aruco_detect(padded)
        if debug:
            ids_found = det_ids.flatten().tolist() if det_ids is not None else []
            print(f"[ArUco] Marker {mid} warp — detected IDs: {ids_found}")
        if det_ids is None:
            continue

        for i, det_id in enumerate(det_ids.flatten()):
            if int(det_id) != mid:
                continue   # expect only this marker's ID

            # ── map detected corners: padded → warp → original image ──────
            c_pad  = det_corners[i][0].astype(np.float32)        # (4,2) padded
            c_warp = c_pad - np.float32([PAD, PAD])               # remove pad offset
            # Homogeneous inverse perspective transform
            M_inv  = np.linalg.inv(M)
            c_h    = np.column_stack([c_warp, np.ones(4, dtype=np.float32)])
            c_orig = (M_inv @ c_h.T).T
            c_orig = (c_orig[:, :2] / c_orig[:, 2:3]).astype(np.float32)

            obj_pts_list.append(world_corners)
            img_pts_list.append(c_orig)
            detected_ids.append(mid)
            break   # one detection per marker

    if not obj_pts_list:
        print("[ArUco] No markers detected.")
        return None, None

    obj_pts = np.vstack(obj_pts_list).astype(np.float32)
    img_pts = np.vstack(img_pts_list).astype(np.float32)

    # ── solvePnP ─────────────────────────────────────────────────────────────
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("[ArUco] solvePnP failed.")
        return None, None

    R_wc, _ = cv2.Rodrigues(rvec)
    # solvePnP returns OpenCV convention (cam looks +Z, Y down).
    # MuJoCo convention (cam looks -Z, Y up) differs by C = diag(1,-1,-1).
    # R_cam_mujoco = R_wc.T @ C  maps MuJoCo cam frame → world.
    _C      = np.diag([1., -1., -1.])
    R_cam   = R_wc.T @ _C
    t_cam   = -R_wc.T @ tvec.flatten()

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))

    if debug:
        print(f"[ArUco] Detected IDs       : {detected_ids}")
        print(f"[ArUco] Camera pos (world) : {np.round(t_cam, 4)}")
        print(f"[ArUco] Reprojection error : {reproj_err:.3f} px")

    if reproj_err > 15.0:
        print(f"[ArUco] WARNING: reprojection error {reproj_err:.1f} px — "
              "check corner ordering in ARUCO_MARKER_WORLD_CORNERS.")

    return R_cam, t_cam


def localize_block_aruco(image_bgr, R_cam_nominal, t_cam_nominal,
                          fovy=120, img_w=640, img_h=480):
    """
    Full pipeline: ArUco → camera pose → block localization.

    Parameters
    ----------
    R_cam_nominal, t_cam_nominal : nominal camera extrinsics from MuJoCo
        (data.cam_xmat reshaped to (3,3) and data.cam_xpos)

    Returns
    -------
    p_world      : (3,) block centre in world frame, or None
    p_robot_base : (3,) block centre in robot-base frame, or None
    """
    R_cam, t_cam = estimate_camera_pose_aruco(
        image_bgr, R_cam_nominal, t_cam_nominal, fovy, img_w, img_h
    )
    if R_cam is None:
        print("[Localize] ArUco pose estimation failed.")
        return None, None

    result = detect_block_2d(image_bgr)
    if result is None:
        print("[Localize] Block not found in image.")
        return None, None

    cx, cy, w, h = result
    # Bottom-centre of bounding box = where block base meets floor
    p_world = pixel_to_world_ground(cx, cy + h // 2,
                                    R_cam, t_cam,
                                    fovy_deg=fovy,
                                    img_w=img_w, img_h=img_h,
                                    ground_z=0.0)
    if p_world is None:
        print("[Localize] Ground-plane intersection failed.")
        return None, None

    p_world[2] = 0.09          # block centre height (half of 0.18 m box)
    p_robot_base = p_world - ROBOT_BASE_WORLD_POS

    print(f"[Localize] Block — world frame      : {np.round(p_world,      4)}")
    print(f"[Localize] Block — robot-base frame : {np.round(p_robot_base, 4)}")
    return p_world, p_robot_base