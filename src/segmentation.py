
import cv2
import numpy as np


def segment_lines(image):
    """
    Detect and extract individual handwritten text lines.
    Accepts a PIL Image and returns a list of line images.
    """

    # Convert PIL image to OpenCV format
    image = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Convert text into white pixels and background into black
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Count text pixels in each horizontal row
    horizontal_projection = np.sum(binary, axis=1)

    # Identify rows containing text
    threshold = max(1000, np.max(horizontal_projection) * 0.03)
    text_rows = horizontal_projection > threshold

    # Group consecutive text rows into lines
    lines = []
    start = None

    for y, has_text in enumerate(text_rows):
        if has_text and start is None:
            start = y

        elif not has_text and start is not None:
            end = y
            if end - start >= 5:
                lines.append((start, end))
            start = None

    if start is not None:
        lines.append((start, len(text_rows)))

    # Crop detected lines with horizontal and vertical padding
    cropped_lines = []

    for start, end in lines:
        padding = 10

        top = max(0, start - padding)
        bottom = min(image.shape[0], end + padding)

        line_binary = binary[top:bottom, :]

        # Find columns containing actual handwriting
        column_projection = np.sum(line_binary, axis=0)

        active_columns = np.where(
            column_projection > max(2, (bottom - top) * 0.03)
        )[0]

        if len(active_columns) == 0:
            continue

        left = max(0, active_columns[0] - 15)
        right = min(image.shape[1], active_columns[-1] + 15)

        cropped = image[top:bottom, left:right]
        cropped_lines.append(cropped)

    return cropped_lines
