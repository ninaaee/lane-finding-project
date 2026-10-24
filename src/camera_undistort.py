
import cv2 as cv
import numpy as np
import os

from camera_utils import undistort_for_image
from config import IMAGE_NAME

# Load previously calculated camera calibration parameters
calibration = np.load("output_images/calibration.npz")

mtx = calibration["mtx"]
dist = calibration["dist"]
calibration_size = tuple(calibration["image_size"])

# Load calibration image
#image = cv.imread("camera_cal/calibration2.jpg")
image = cv.imread(f"test_images/{IMAGE_NAME}.jpg")

if image is None:
    raise RuntimeError("Failed to load calibration image.")

# Correct lens distortion
undistorted = undistort_for_image(
    image, mtx, dist, calibration_size
)

# Compare original and corrected images
comparison = np.hstack((image, undistorted))

# Resize comparison for display only
display = cv.resize(
    comparison,
    None,
    fx=0.5,
    fy=0.5,
    interpolation=cv.INTER_AREA
)

cv.imshow("Original vs Undistorted", display)

# Save result
os.makedirs("output_images", exist_ok=True)

cv.imwrite(
    f"output_images/{IMAGE_NAME}_undistorted.jpg",
    comparison
)

cv.waitKey(0)
cv.destroyAllWindows()
