
import cv2 as cv
import numpy as np
import os

from camera_utils import undistort_for_image
from config import APPLY_ROI, IMAGE_NAME, THRESHOLD_METHOD
from thresholding import threshold_image


def get_perspective_matrices(width, height):
    """
    Calculate forward and inverse perspective matrices.

    Returns:
        M: Original perspective to bird's-eye view.
        M_inv: Bird's-eye view to original perspective.
    """

    # Reference points for 1280x720 images
    reference_width = 1280
    reference_height = 720

    # Scale the points to the current image dimensions
    scale_x = width / reference_width
    scale_y = height / reference_height

    # Source points selected from straight_lines1.jpg
    src = np.float32([
        [279 * scale_x, 674 * scale_y],   # Bottom-left
        [602 * scale_x, 446 * scale_y],   # Top-left
        [676 * scale_x, 446 * scale_y],   # Top-right
        [1031 * scale_x, 674 * scale_y]   # Bottom-right
    ])


    # Destination points in matching order
    dst = np.float32([
        [0.25 * width, height - 1],
        [0.25 * width, 0],
        [0.75 * width, 0],
        [0.75 * width, height - 1]
    ])

    # Calculate perspective transformation matrices
    M = cv.getPerspectiveTransform(src, dst)
    M_inv = cv.getPerspectiveTransform(dst, src)

    return M, M_inv, src


def warp_image(image, M, binary=False):
    """
    Apply perspective transformation to an image.
    """

    height, width = image.shape[:2]

    # Use nearest-neighbor interpolation for binary masks
    interpolation = (
        cv.INTER_NEAREST if binary else cv.INTER_LINEAR
    )

    warped = cv.warpPerspective(
        image,
        M,
        (width, height),
        flags=interpolation
    )

    return warped


if __name__ == "__main__":

    # --------------------------------------------
    # 1. Load calibration parameters
    # --------------------------------------------

    calibration = np.load("output_images/calibration.npz")

    mtx = calibration["mtx"]
    dist = calibration["dist"]
    calibration_size = tuple(calibration["image_size"])

    # --------------------------------------------
    # 2. Load test image
    # --------------------------------------------

    image = cv.imread(f"test_images/{IMAGE_NAME}.jpg")

    if image is None:
        raise RuntimeError("Failed to load test image.")

    height, width = image.shape[:2]

    # --------------------------------------------
    # 3. Correct lens distortion
    # --------------------------------------------

    undistorted = undistort_for_image(
        image, mtx, dist, calibration_size
    )

    # --------------------------------------------
    # 4. Apply binary thresholding
    # --------------------------------------------

    binary = threshold_image(
        undistorted,
        method=THRESHOLD_METHOD,
        apply_roi=APPLY_ROI
    )

    # --------------------------------------------
    # 5. Calculate perspective matrices
    # --------------------------------------------

    M, M_inv, src = get_perspective_matrices(
        width, height
    )

    # --------------------------------------------
    # 6. Transform original and binary images
    # --------------------------------------------

    warped_color = warp_image(
        undistorted,
        M
    )

    warped_binary = warp_image(
        binary,
        M,
        binary=True
    )

    # --------------------------------------------
    # 7. Draw source points
    # --------------------------------------------

    preview = undistorted.copy()

    cv.polylines(
        preview,
        [src.astype(np.int32)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3
    )

    # --------------------------------------------
    # 8. Save results
    # --------------------------------------------

    os.makedirs("output_images", exist_ok=True)

    cv.imwrite(
        f"output_images/{IMAGE_NAME}_perspective_source.png",
        preview
    )

    cv.imwrite(
        f"output_images/{IMAGE_NAME}_warped_color.png",
        warped_color
    )

    cv.imwrite(
        f"output_images/{IMAGE_NAME}_warped_binary.png",
        warped_binary * 255
    )

    print("Perspective transformation results saved.")

    # --------------------------------------------
    # 9. Display results
    # --------------------------------------------

    cv.namedWindow("Source Region", cv.WINDOW_NORMAL)
    cv.imshow("Source Region", preview)

    cv.namedWindow("Bird's-eye View", cv.WINDOW_NORMAL)
    cv.imshow("Bird's-eye View", warped_color)

    cv.namedWindow("Warped Binary", cv.WINDOW_NORMAL)
    cv.imshow("Warped Binary", warped_binary * 255)

    cv.waitKey(0)
    cv.destroyAllWindows()
