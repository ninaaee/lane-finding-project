
import cv2 as cv
import numpy as np

from camera_utils import undistort_for_image
from config import APPLY_ROI, IMAGE_NAME, THRESHOLD_METHOD
from thresholding import threshold_image
from perspective_transform import (
    get_perspective_matrices,
    warp_image
)
from lane_detection import detect_lane_lines


def draw_lane(undistorted, left_fit, right_fit, inverse_matrix):
    """Draw the detected lane area on an undistorted road image."""

    height, width = undistorted.shape[:2]

    # Generate vertical positions across the image
    plot_y = np.linspace(0, height - 1, height)

    # Calculate horizontal lane positions
    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    # Create an empty color image
    lane_image = np.zeros_like(undistorted)

    # Prepare coordinates for the lane polygon
    left_points = np.column_stack((left_fit_x, plot_y))
    right_points = np.column_stack((right_fit_x, plot_y))[::-1]

    lane_points = np.vstack((left_points, right_points))
    lane_points = np.round(lane_points).astype(np.int32)

    # Fill the lane area with green
    cv.fillPoly(
        lane_image,
        [lane_points],
        (0, 255, 0)
    )

    # Transform the lane area back to the original perspective
    unwarped_lane = cv.warpPerspective(
        lane_image,
        inverse_matrix,
        (width, height),
        flags=cv.INTER_LINEAR
    )

    # Overlay the lane area on the original image
    result = cv.addWeighted(
        undistorted,
        1.0,
        unwarped_lane,
        0.3,
        0
    )

    return result


def add_lane_measurements(image, detection):
    """Display lane curvature and vehicle offset."""

    result = image.copy()

    left_curvature = detection["left_curvature"]
    right_curvature = detection["right_curvature"]
    vehicle_offset = detection["vehicle_offset"]

    average_curvature = (
        left_curvature + right_curvature
    ) / 2

    if vehicle_offset > 0:
        direction = "right"
    elif vehicle_offset < 0:
        direction = "left"
    else:
        direction = "center"

    curvature_text = (
        f"Lane curvature: {average_curvature:.0f} m"
    )

    offset_text = (
        f"Vehicle offset: {abs(vehicle_offset):.2f} m {direction}"
    )

    cv.putText(
        result,
        curvature_text,
        (40, 65),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    cv.putText(
        result,
        offset_text,
        (40, 115),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv.LINE_AA
    )

    return result


def main():
    # Load calibration parameters
    calibration = np.load("output_images/calibration.npz")
    mtx = calibration["mtx"]
    dist = calibration["dist"]
    calibration_size = tuple(calibration["image_size"])

    # Load original road image
    original = cv.imread(
        f"test_images/{IMAGE_NAME}.jpg"
    )

    if original is None:
        raise RuntimeError("Failed to load road image.")

    # Correct lens distortion
    undistorted = undistort_for_image(
        original, mtx, dist, calibration_size
    )

    # Apply combined thresholding
    binary = threshold_image(
        undistorted,
        method=THRESHOLD_METHOD,
        apply_roi=APPLY_ROI
    )

    # Calculate perspective transformation matrices
    height, width = binary.shape

    matrix, inverse_matrix, _ = get_perspective_matrices(
        width,
        height
    )

    # Transform binary image to bird's-eye view
    binary_warped = warp_image(
        binary,
        matrix,
        binary=True
    )

    # Detect lane lines
    detection = detect_lane_lines(
        binary_warped,
        visualize=True
    )

    # Draw detected lane area
    lane_result = draw_lane(
        undistorted,
        detection["left_fit"],
        detection["right_fit"],
        inverse_matrix
    )

    # Add curvature and offset information
    final_result = add_lane_measurements(
        lane_result,
        detection
    )

    # Save final visualization
    output_path = (
        f"output_images/{IMAGE_NAME}_lane_overlay.png"
    )

    cv.imwrite(output_path, final_result)
    print(f"Saved: {output_path}")

    # Display the result
    cv.namedWindow("Lane Visualization", cv.WINDOW_NORMAL)
    cv.imshow("Lane Visualization", final_result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
