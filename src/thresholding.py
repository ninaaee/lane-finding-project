
import cv2 as cv
import numpy as np
import os


def threshold_image(undistorted, method="color"):
    """
    Extract lane marking candidates from an undistorted image.

    Args:
        undistorted: Input BGR image.
        method: "color", "sobel", or "combined".

    Returns:
        Binary image containing values 0 and 1.
    """

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
    white_binary = np.zeros_like(l_channel)

    white_binary[
        (l_channel >= 200) &
        (s_channel <= 120)
    ] = 1

    # Detect yellow lane markings
    yellow_binary = np.zeros_like(h_channel)

    yellow_binary[
        (h_channel >= 15) &
        (h_channel <= 40) &
        (s_channel >= 80) &
        (l_channel >= 40)
    ] = 1

    # Combine white and yellow masks
    color_binary = np.zeros_like(l_channel)

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
        binary = np.zeros_like(color_binary)

        binary[
            (sobel_binary == 1) |
            (color_binary == 1)
        ] = 1

    else:
        raise ValueError(
            f"Unknown thresholding method: {method}"
        )

    # --------------------------------------------
    # 4. Region of Interest
    # --------------------------------------------

    height, width = binary.shape

    roi_mask = np.zeros_like(binary)

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

    # Load calibration parameters
    calibration = np.load("output_images/calibration.npz")

    mtx = calibration["mtx"]
    dist = calibration["dist"]

    # Load test image
    image = cv.imread("test_images/test1.jpg")

    if image is None:
        raise RuntimeError("Failed to load test image.")

    # Check image resolution
    image_size = (image.shape[1], image.shape[0])
    calibration_size = tuple(calibration["image_size"])

    if image_size != calibration_size:
        raise RuntimeError(
            "Image resolution does not match calibration."
        )

    # Correct lens distortion
    undistorted = cv.undistort(image, mtx, dist, None, mtx)

    # Apply thresholding
    final_binary = threshold_image(
        undistorted,
        method="color"
    )

    # Save result
    os.makedirs("output_images", exist_ok=True)

    cv.imwrite(
        "output_images/color_roi_test1.png",
        final_binary * 255
    )

    # Display result
    cv.namedWindow("Color Threshold + ROI", cv.WINDOW_NORMAL)
    cv.imshow("Color Threshold + ROI", final_binary * 255)

    cv.waitKey(0)
    cv.destroyAllWindows()
