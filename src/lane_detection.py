
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import IMAGE_NAME


# --------------------------------------------
# Configuration
# --------------------------------------------

N_WINDOWS = 9
MARGIN = 100
MIN_PIXELS = 50

# --------------------------------------------
# 1. Histogram and Sliding Windows
# --------------------------------------------

def find_lane_pixels(
    binary_warped,
    visualize=True,
    margin=None,
    min_pixels=None
):
    """
    Find lane pixels using histogram and sliding windows.

    Args:
        binary_warped: Bird's-eye binary image.
        visualize: Whether to draw sliding windows.

    Returns:
        left_x, left_y: Left lane pixel coordinates.
        right_x, right_y: Right lane pixel coordinates.
        output_image: Visualization of detected pixels.
        histogram: Histogram of the bottom half.
        left_base, right_base: Initial lane positions.
    """

    if binary_warped is None or binary_warped.ndim != 2:
        raise ValueError("Expected a two-dimensional binary image.")

    binary = (binary_warped > 0).astype(np.uint8)

    height, width = binary.shape
    margin = margin or max(1, round(MARGIN * width / 1280))
    min_pixels = min_pixels or max(
        1,
        int(MIN_PIXELS * width * height / (1280 * 720))
    )

    # Calculate histogram of the bottom half
    histogram = np.sum(binary[height // 2:, :], axis=0)

    # Find initial lane positions
    midpoint = width // 2

    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    if histogram[left_base] == 0 or histogram[right_base] == 0:
        raise RuntimeError("Could not find both lane bases.")

    # Sliding window parameters
    window_height = height // N_WINDOWS

    nonzero_y, nonzero_x = binary.nonzero()

    left_current = left_base
    right_current = right_base

    left_lane_indices = []
    right_lane_indices = []

    # Create visualization image
    output_image = np.dstack((binary, binary, binary)) * 255

    # Search from bottom to top
    for window in range(N_WINDOWS):

        # Vertical window boundaries
        y_low = height - (window + 1) * window_height

        # Include the topmost rows in the final window
        if window == N_WINDOWS - 1:
            y_low = 0

        y_high = height - window * window_height

        # Horizontal window boundaries
        left_x_low = left_current - margin
        left_x_high = left_current + margin

        right_x_low = right_current - margin
        right_x_high = right_current + margin

        # Draw sliding windows
        if visualize:
            cv.rectangle(
                output_image,
                (left_x_low, y_low),
                (left_x_high, y_high),
                (0, 255, 0),
                2
            )

            cv.rectangle(
                output_image,
                (right_x_low, y_low),
                (right_x_high, y_high),
                (0, 255, 0),
                2
            )

        # Find pixels inside the left window
        good_left = (
            (nonzero_y >= y_low) &
            (nonzero_y < y_high) &
            (nonzero_x >= left_x_low) &
            (nonzero_x < left_x_high)
        ).nonzero()[0]

        # Find pixels inside the right window
        good_right = (
            (nonzero_y >= y_low) &
            (nonzero_y < y_high) &
            (nonzero_x >= right_x_low) &
            (nonzero_x < right_x_high)
        ).nonzero()[0]

        left_lane_indices.append(good_left)
        right_lane_indices.append(good_right)

        # Recenter windows when enough pixels are found
        if len(good_left) > min_pixels:
            left_current = int(np.mean(nonzero_x[good_left]))

        if len(good_right) > min_pixels:
            right_current = int(np.mean(nonzero_x[good_right]))

    # Combine detected indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract detected pixel coordinates
    left_x = nonzero_x[left_lane_indices]
    left_y = nonzero_y[left_lane_indices]

    right_x = nonzero_x[right_lane_indices]
    right_y = nonzero_y[right_lane_indices]

    if len(left_x) < 3 or len(right_x) < 3:
        raise RuntimeError("Not enough lane pixels for polynomial fitting.")

    # Highlight detected lane pixels
    if visualize:
        output_image[left_y, left_x] = [255, 0, 0]
        output_image[right_y, right_x] = [0, 0, 255]

    return (
        left_x,
        left_y,
        right_x,
        right_y,
        output_image,
        histogram,
        left_base,
        right_base
    )


# --------------------------------------------
# 2. Polynomial Fitting
# --------------------------------------------

def fit_lane_polynomials(left_x, left_y, right_x, right_y, height):
    """
    Fit second-degree polynomials to lane pixels.

    Returns:
        left_fit, right_fit: Polynomial coefficients.
        plot_y: Vertical image coordinates.
        left_fit_x, right_fit_x: Fitted horizontal coordinates.
    """

    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    plot_y = np.linspace(0, height - 1, height)

    left_fit_x = np.polyval(left_fit, plot_y)
    right_fit_x = np.polyval(right_fit, plot_y)

    return left_fit, right_fit, plot_y, left_fit_x, right_fit_x


# --------------------------------------------
# 3. Radius of Curvature
# --------------------------------------------

def calculate_curvature(
    left_x,
    left_y,
    right_x,
    right_y,
    image_height,
    image_width
):
    """Calculate the approximate curvature radius in meters."""

    ym_per_pix = 30 / image_height
    xm_per_pix = 3.7 / (image_width * 0.5)

    # Fit polynomials in real-world coordinates
    left_fit_m = np.polyfit(
        left_y * ym_per_pix,
        left_x * xm_per_pix,
        2
    )

    right_fit_m = np.polyfit(
        right_y * ym_per_pix,
        right_x * xm_per_pix,
        2
    )

    # Evaluate curvature at the bottom
    y_eval = (image_height - 1) * ym_per_pix

    def curvature_radius(fit):
        a, b, _ = fit

        if abs(a) < 1e-12:
            return float("inf")

        return (
            (1 + (2 * a * y_eval + b) ** 2) ** 1.5
            / abs(2 * a)
        )

    return curvature_radius(left_fit_m), curvature_radius(right_fit_m)


# --------------------------------------------
# 4. Vehicle Offset
# --------------------------------------------

def calculate_vehicle_offset(left_fit, right_fit, image_width, image_height):
    """Estimate vehicle offset from the lane center in meters."""

    y_eval = image_height - 1

    # Estimate lane boundary positions at the bottom
    left_position = np.polyval(left_fit, y_eval)
    right_position = np.polyval(right_fit, y_eval)

    lane_center = (left_position + right_position) / 2

    # Assume that the camera is mounted at the vehicle center
    vehicle_center = image_width / 2

    # Positive offset means the vehicle is right of lane center
    xm_per_pix = 3.7 / (image_width * 0.5)
    offset_m = (vehicle_center - lane_center) * xm_per_pix

    return offset_m


# --------------------------------------------
# 5. Lane Detection
# --------------------------------------------

def detect_lane_lines(binary_warped, visualize=True):
    """
    Detect lane boundaries and calculate lane measurements.

    Args:
        binary_warped: Binary bird's-eye view image.
        visualize: Whether to create a detection visualization.

    Returns:
        Dictionary containing polynomial fits, measurements,
        histogram, and visualization.
    """

    binary = (binary_warped > 0).astype(np.uint8)
    height, width = binary.shape

    (
        left_x,
        left_y,
        right_x,
        right_y,
        output_image,
        histogram,
        left_base,
        right_base
    ) = find_lane_pixels(
        binary,
        visualize=visualize
    )

    # Fit lane polynomials
    (
        left_fit,
        right_fit,
        plot_y,
        left_fit_x,
        right_fit_x
    ) = fit_lane_polynomials(
        left_x,
        left_y,
        right_x,
        right_y,
        height
    )

    # Calculate curvature
    left_curvature, right_curvature = calculate_curvature(
        left_x,
        left_y,
        right_x,
        right_y,
        height,
        width
    )

    # Calculate vehicle offset
    vehicle_offset = calculate_vehicle_offset(
        left_fit,
        right_fit,
        width,
        height
    )

    # Calculate lane width
    lane_width_bottom = right_fit_x[-1] - left_fit_x[-1]
    lane_width_top = right_fit_x[0] - left_fit_x[0]

    # Draw fitted curves
    if visualize:
        left_points = np.column_stack((left_fit_x, plot_y))
        right_points = np.column_stack((right_fit_x, plot_y))

        cv.polylines(
            output_image,
            [np.round(left_points).astype(np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=4
        )

        cv.polylines(
            output_image,
            [np.round(right_points).astype(np.int32)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=4
        )

    return {
        "left_fit": left_fit,
        "right_fit": right_fit,
        "plot_y": plot_y,
        "left_fit_x": left_fit_x,
        "right_fit_x": right_fit_x,
        "left_curvature": left_curvature,
        "right_curvature": right_curvature,
        "vehicle_offset": vehicle_offset,
        "lane_width_bottom": lane_width_bottom,
        "lane_width_top": lane_width_top,
        "left_base": left_base,
        "right_base": right_base,
        "histogram": histogram,
        "visualization": output_image
    }


# --------------------------------------------
# 6. Main - Test Lane Detection
# --------------------------------------------

def main():
    # Select the warped binary image to test
    input_path = Path(
        f"output_images/{IMAGE_NAME}_warped_binary.png"
    )

    output_path = Path(
        f"output_images/{IMAGE_NAME}_lane_detection.png"
    )

    # Load binary bird's-eye view image
    image = cv.imread(
        str(input_path),
        cv.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    # Perform lane detection
    result = detect_lane_lines(image, visualize=True)

    # Print results
    print(f"Left lane base: {result['left_base']}")
    print(f"Right lane base: {result['right_base']}")

    print("Left lane polynomial:", result["left_fit"])
    print("Right lane polynomial:", result["right_fit"])

    print(
        f"Left curvature: {result['left_curvature']:.2f} m"
    )
    print(
        f"Right curvature: {result['right_curvature']:.2f} m"
    )

    print(
        f"Vehicle offset: {result['vehicle_offset']:+.3f} m"
    )

    print(
        f"Lane width at bottom: "
        f"{result['lane_width_bottom']:.2f} px"
    )
    print(
        f"Lane width at top: "
        f"{result['lane_width_top']:.2f} px"
    )

    # Save visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv.imwrite(
        str(output_path),
        result["visualization"]
    )

    print(f"Saved: {output_path}")

    # Display lane detection result
    cv.namedWindow("Lane Detection", cv.WINDOW_NORMAL)

    cv.imshow(
        "Lane Detection",
        result["visualization"]
    )

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Display histogram
    plt.figure(figsize=(12, 5))

    plt.plot(
        result["histogram"],
        label="Pixel histogram"
    )

    plt.axvline(
        result["left_base"],
        color="red",
        linestyle="--",
        label="Left lane base"
    )

    plt.axvline(
        result["right_base"],
        color="green",
        linestyle="--",
        label="Right lane base"
    )

    plt.xlabel("Horizontal pixel position (x)")
    plt.ylabel("Number of detected pixels")
    plt.title("Lane Pixel Histogram")

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
