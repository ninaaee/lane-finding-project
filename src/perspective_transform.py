
import cv2 as cv
import numpy as np
import os

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
        [230 * scale_x, 700 * scale_y],
        [575 * scale_x, 455 * scale_y],
        [710 * scale_x, 455 * scale_y],
        [1050 * scale_x, 700 * scale_y]
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

    # --------------------------------------------
    # 2. Load test image
    # --------------------------------------------

    image_name = "straight_lines1"

    image = cv.imread(f"test_images/{image_name}.jpg")

    if image is None:
        raise RuntimeError("Failed to load test image.")

    height, width = image.shape[:2]

    calibration_size = tuple(calibration["image_size"])

    if (width, height) != calibration_size:
        raise RuntimeError(
            "Image resolution does not match calibration."
        )

    # --------------------------------------------
    # 3. Correct lens distortion
    # --------------------------------------------

    undistorted = cv.undistort(
        image, mtx, dist, None, mtx
    )

    # --------------------------------------------
    # 4. Apply binary thresholding
    # --------------------------------------------

    binary = threshold_image(
        undistorted,
        method="color"
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
        f"output_images/{image_name}_perspective_source.png",
        preview
    )

    cv.imwrite(
        f"output_images/{image_name}_warped_color.png",
        warped_color
    )

    cv.imwrite(
        f"output_images/{image_name}_warped_binary.png",
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
