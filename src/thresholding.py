
import cv2 as cv
import numpy as np
import os

from camera_utils import undistort_for_image
from config import IMAGE_NAME


def threshold_image(undistorted, method="color", apply_roi=True):
    """
    Extract lane marking candidates from an undistorted image.

    Args:
        undistorted: Input BGR image.
        method: "color", "sobel", or "combined".
        apply_roi: Whether to apply a trapezoidal region of interest.

    Returns:
        Binary image containing values 0 and 1.
    """

    if undistorted is None or undistorted.size == 0:
        raise ValueError("Input image must not be empty.")

    # --------------------------------------------
    # 1. Sobel X thresholding
    # --------------------------------------------

    gray = cv.cvtColor(undistorted, cv.COLOR_BGR2GRAY)

    sobel_x = cv.Sobel(
        gray, cv.CV_64F, 1, 0, ksize=3
    )

    abs_sobel_x = np.absolute(sobel_x)
    max_gradient = np.max(abs_sobel_x)

    if max_gradient > 0:
        scaled_sobel = np.uint8(
            255 * abs_sobel_x / max_gradient
        )
    else:
        scaled_sobel = np.zeros_like(gray, dtype=np.uint8)

    sobel_binary = np.zeros_like(gray, dtype=np.uint8)

    sobel_binary[
        (scaled_sobel >= 20) &
        (scaled_sobel <= 100)
    ] = 1

    # --------------------------------------------
    # 2. HLS color thresholding
    # --------------------------------------------

    hls = cv.cvtColor(undistorted, cv.COLOR_BGR2HLS)

    h_channel = hls[:, :, 0]
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Detect white lane markings
    white_binary = np.zeros_like(l_channel, dtype=np.uint8)

    white_binary[
        (l_channel >= 200) &
        (s_channel <= 120)
    ] = 1

    # Detect yellow lane markings
    yellow_binary = np.zeros_like(h_channel, dtype=np.uint8)

    yellow_binary[
        (h_channel >= 15) &
        (h_channel <= 40) &
        (s_channel >= 80) &
        (l_channel >= 40)
    ] = 1

    # Combine white and yellow masks
    color_binary = np.zeros_like(l_channel, dtype=np.uint8)

    color_binary[
        (white_binary == 1) |
        (yellow_binary == 1)
    ] = 1

    # --------------------------------------------
    # 3. Select thresholding method
    # --------------------------------------------

    if method == "color":
        binary = color_binary

    elif method == "sobel":
        binary = sobel_binary

    elif method == "combined":
        binary = np.logical_or(
            sobel_binary,
            color_binary
        ).astype(np.uint8)

    else:
        raise ValueError(
            f"Unknown thresholding method: {method}"
        )

    # --------------------------------------------
    # 4. Optional Region of Interest
    # --------------------------------------------

    if not apply_roi:
        return binary

    height, width = binary.shape

    roi_mask = np.zeros_like(binary, dtype=np.uint8)

    vertices = np.array([[
        (int(width * 0.05), height - 1),
        (int(width * 0.40), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.55)),
        (int(width * 0.95), height - 1)
    ]], dtype=np.int32)

    cv.fillPoly(roi_mask, vertices, 1)

    final_binary = cv.bitwise_and(binary, roi_mask)

    return final_binary


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

    image = cv.imread(
        f"test_images/{IMAGE_NAME}.jpg"
    )

    if image is None:
        raise RuntimeError("Failed to load test image.")

    # --------------------------------------------
    # 3. Correct lens distortion
    # --------------------------------------------

    undistorted = undistort_for_image(
        image, mtx, dist, calibration_size
    )

    # --------------------------------------------
    # 4. Apply different thresholding methods
    # --------------------------------------------

    color_binary = threshold_image(
        undistorted,
        method="color",
        apply_roi=True
    )

    sobel_binary = threshold_image(
        undistorted,
        method="sobel",
        apply_roi=True
    )

    combined_binary = threshold_image(
        undistorted,
        method="combined",
        apply_roi=True
    )

    # --------------------------------------------
    # 5. Save results
    # --------------------------------------------

    os.makedirs("output_images", exist_ok=True)

    color_path = f"output_images/{IMAGE_NAME}_color.png"
    sobel_path = f"output_images/{IMAGE_NAME}_sobel.png"
    combined_path = f"output_images/{IMAGE_NAME}_combined.png"

    cv.imwrite(color_path, color_binary * 255)
    cv.imwrite(sobel_path, sobel_binary * 255)
    cv.imwrite(combined_path, combined_binary * 255)

    print(f"Saved: {color_path}")
    print(f"Saved: {sobel_path}")
    print(f"Saved: {combined_path}")

    # --------------------------------------------
    # 6. Display comparison
    # --------------------------------------------

    comparison = np.hstack((
        color_binary * 255,
        sobel_binary * 255,
        combined_binary * 255
    ))

    cv.namedWindow(
        "Color | Sobel X | Combined",
        cv.WINDOW_NORMAL
    )

    cv.imshow(
        "Color | Sobel X | Combined",
        comparison
    )

    cv.waitKey(0)
    cv.destroyAllWindows()
