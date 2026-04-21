"""
Generate ArUco marker PNG files for embedding in the MuJoCo scene.
Markers are from DICT_4X4_50. IDs 0 and 1 are placed on the floor
at the left and right sides of the robot.
"""

import cv2
import numpy as np

# Works with both OpenCV <4.7 (drawMarker) and >=4.7 (generateImageMarker)
def generate_marker(dictionary, marker_id: int, size_px: int = 200) -> np.ndarray:
    try:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    except AttributeError:
        img = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size_px, img)
    # Add a white border (10% of size) so the marker is readable when textured
    border = size_px // 10
    canvas = np.ones((size_px + 2 * border, size_px + 2 * border), dtype=np.uint8) * 255
    canvas[border:border + size_px, border:border + size_px] = img
    # Convert to BGR so cv2.imwrite saves a standard PNG
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # ID 1 has adjacent all-black data rows which confuse the border detector;
    # use IDs 0 and 2 instead.  aruco_marker_1.png holds ID=2 content.
    for marker_id in [0, 2]:
        img = generate_marker(dictionary, marker_id, size_px=512)
        file_idx = 0 if marker_id == 0 else 1   # aruco_marker_1.png holds ID=2
        path = f"aruco_marker_{file_idx}.png"
        cv2.imwrite(path, img)
        print(f"Saved {path}  ({img.shape[1]}x{img.shape[0]} px)")
