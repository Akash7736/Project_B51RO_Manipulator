

import cv2
import numpy as np

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