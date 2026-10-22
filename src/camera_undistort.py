
import cv2 as cv
import numpy as np
import os

# Load previously calculated camera calibration parameters
calibration = np.load("output_images/calibration.npz")

mtx = calibration["mtx"]
dist = calibration["dist"]

# Load calibration image
image = cv.imread("camera_cal/calibration2.jpg")
#image = cv.imread("test_images/test1.jpg")

if image is None:
    raise RuntimeError("Failed to load calibration image.")

# Correct lens distortion
undistorted = cv.undistort(image, mtx, dist, None, mtx)

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
    "output_images/undistorted_chessboard.jpg",
    comparison
)

cv.waitKey(0)
cv.destroyAllWindows()
