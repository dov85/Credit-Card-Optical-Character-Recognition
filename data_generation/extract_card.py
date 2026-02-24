import cv2
import numpy as np
import json
import os
import glob


def order_points(pts):
    """
    Sorts coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # Top-Left
    rect[2] = pts[np.argmax(s)]    # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def get_card_perspective_matrix(image, quad_coords, output_size=None):
    """
    Computes the perspective matrix that warps the card region
    (defined by quad_coords) to fill the entire output image.

    Args:
        image: Input image (numpy array) or image path (str).
        quad_coords: 4 corner points of the card in the image.
                     Can be a list/array of shape (4, 2), or a path
                     to a JSON file with a "quad" key.
        output_size: (width, height) of the output. If None, uses
                     the standard credit card aspect ratio (85.6 x 53.98 mm)
                     scaled to a reasonable resolution.

    Returns:
        matrix: The 3x3 perspective transformation matrix.
        warped: The warped image (card filling the entire frame).
    """
    # Load image if path
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image}")

    # Load coordinates from JSON if path
    if isinstance(quad_coords, str):
        with open(quad_coords, 'r') as f:
            data = json.load(f)
        quad_coords = np.array(data["quad"], dtype="float32")
    else:
        quad_coords = np.array(quad_coords, dtype="float32")

    # Order the source points
    src_pts = order_points(quad_coords)

    # Determine output size
    if output_size is None:
        # Credit card ratio: 85.6 x 53.98 mm ≈ 1.586
        # Use a reasonable resolution
        out_w = 512
        out_h = int(out_w / 1.586)  # ~323
    else:
        out_w, out_h = output_size

    # Destination points: full image rectangle
    dst_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the warp
    warped = cv2.warpPerspective(image, matrix, (out_w, out_h))

    return matrix, warped


def process_all(image_dir, json_dir, output_dir=None):
    """
    Process all image-JSON pairs from temp_extract and extract
    the card region from each.

    Args:
        image_dir: Path to folder with background images.
        json_dir: Path to folder with corresponding JSON files.
        output_dir: Optional path to save extracted card images.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))

    for img_path in sorted(image_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(json_dir, basename + ".json")

        if not os.path.exists(json_path):
            print(f"Skipping {basename}: no JSON found")
            continue

        try:
            matrix, warped = get_card_perspective_matrix(img_path, json_path)
            print(f"{basename}: matrix computed successfully")

            if output_dir:
                save_path = os.path.join(output_dir, f"{basename}_card.png")
                cv2.imwrite(save_path, warped)

        except Exception as e:
            print(f"Error processing {basename}: {e}")


if __name__ == "__main__":
    
    matrix, warped = get_card_perspective_matrix("temp_extract/images/DG/DG01_01.jpg", "temp_extract/ground_truth/DG/DG01_01.json")

    cv2.imshow("Card", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()