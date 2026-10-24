import cv2 as cv
import numpy as np


def undistort_for_image(image, camera_matrix, distortion, calibration_size):
    """Undistort an image using calibration scaled to its resolution."""

    calibration_width, calibration_height = calibration_size
    height, width = image.shape[:2]

    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError("Calibration dimensions must be positive.")

    scale_x = width / calibration_width
    scale_y = height / calibration_height

    image_matrix = camera_matrix.astype(np.float64, copy=True)
    image_matrix[0, :] *= scale_x
    image_matrix[1, :] *= scale_y

    return cv.undistort(
        image,
        image_matrix,
        distortion,
        None,
        image_matrix
    )