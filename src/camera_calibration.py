
import cv2 as cv
import numpy as np
import glob
import os

# Chessboard dimensions (number of inner corners)
chessboard_size = (9, 6)
cols, rows = chessboard_size

# Prepare known 3D coordinates of chessboard corners
objp = np.zeros((cols * rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# Define criteria for subpixel corner refinement
criteria = (
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Load all calibration images
images = sorted(glob.glob("camera_cal/calibration*.jpg"))

if not images:
    raise RuntimeError("No calibration images found.")

image_size = None

for filename in images:
    image = cv.imread(filename)

    if image is None:
        print(f"Failed to load image: {filename}")
        continue

    # Check image resolution without resizing
    current_size = (image.shape[1], image.shape[0])

    if image_size is not None and current_size != image_size:
        print(f"Skipping {filename}: different resolution")
        continue

    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect chessboard corners
    found, corners = cv.findChessboardCorners(
        gray, chessboard_size, None
    )

    if found:
        # Refine corner positions to subpixel accuracy
        corners = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        # Set reference resolution from the first valid image
        if image_size is None:
            image_size = current_size

        # Store corresponding 3D and 2D points
        objpoints.append(objp.copy())
        imgpoints.append(corners)

        print(f"Successfully processed: {filename}")
    else:
        print(f"Chessboard corners not found: {filename}")

if not objpoints:
    raise RuntimeError("No calibration images were processed successfully.")

# Perform camera calibration
rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print(f"\nImages used for calibration: {len(objpoints)}")
print("\nImage resolution:", image_size)
print("\nRMS reprojection error:", rms)
print("\nCamera matrix:\n", mtx)
print("\nDistortion coefficients:\n", dist)

# Save calibration results
os.makedirs("output_images", exist_ok=True)

np.savez(
    "output_images/calibration.npz",
    mtx=mtx,
    dist=dist,
    image_size=np.array(image_size)
)

print("\nCamera calibration results saved successfully.")
