
import cv2 as cv
import numpy as np

from camera_utils import undistort_for_image
from config import IMAGE_NAME

# Load camera calibration parameters
calibration = np.load("output_images/calibration.npz")
mtx = calibration["mtx"]
dist = calibration["dist"]
calibration_size = tuple(calibration["image_size"])

# Load a reference image
image = cv.imread(f"test_images/{IMAGE_NAME}.jpg")

if image is None:
    raise RuntimeError("Failed to load reference image.")

# Correct lens distortion
undistorted = undistort_for_image(
    image, mtx, dist, calibration_size
)

# Store selected points
points = []

# Source point order
point_names = [
    "Top-left",
    "Top-right",
    "Bottom-right",
    "Bottom-left"
]


def mouse_callback(event, x, y, flags, userdata):
    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        print(
            f"{point_names[len(points) - 1]}: ({x}, {y})"
        )


window_name = "Select Perspective Points"

cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.setMouseCallback(window_name, mouse_callback)

while True:
    display = undistorted.copy()

    for i, (x, y) in enumerate(points):
        cv.circle(display, (x, y), 6, (0, 0, 255), -1)

        cv.putText(
            display,
            point_names[i],
            (x + 10, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    if len(points) == 4:
        polygon = np.array(points, dtype=np.int32)

        cv.polylines(
            display,
            [polygon],
            True,
            (0, 255, 0),
            2
        )

    cv.imshow(window_name, display)

    key = cv.waitKey(20) & 0xFF

    if key == ord("r"):
        points.clear()

    elif key == ord("q") or key == 27:
        break

    elif key == 13 and len(points) == 4:
        print("\nSelected source points:")
        print(np.float32(points))
        break

cv.destroyAllWindows()
