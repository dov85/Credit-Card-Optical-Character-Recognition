"""
Overlay synthetic credit cards onto perspective dataset images.

For each of the 1200 synthetic cards, picks a random perspective background,
warps the card into the quad area, and saves the result.
The perspective matrix and quad stay valid since the card region is replaced in-place.
"""

import cv2
import numpy as np
import json
import os
import glob
import random
import shutil


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def overlay_card(bg_img, card_img, dest_coords):
    """
    Warp the synthetic card onto the background at the given quad coordinates.
    Uses perspective transform + mask blending (from creat.py).
    """
    h, w = card_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    dst_pts = order_points(dest_coords)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp card
    warped_card = cv2.warpPerspective(card_img, matrix, (bg_img.shape[1], bg_img.shape[0]))

    # Create and warp mask
    mask_src = np.ones((h, w), dtype="uint8") * 255
    warped_mask = cv2.warpPerspective(mask_src, matrix, (bg_img.shape[1], bg_img.shape[0]))

    # Blend
    mask_inv = cv2.bitwise_not(warped_mask)
    bg_region = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
    card_region = cv2.bitwise_and(warped_card, warped_card, mask=warped_mask)
    result = cv2.add(bg_region, card_region)
    return result


def main():
    # --- Paths ---
    SYNTHETIC_CARDS_DIR = "dataset/images"
    SYNTHETIC_LABELS_DIR = "dataset/labels"
    PERSPECTIVE_DIR = "perspective_dataset"
    PERSPECTIVE_IMAGES_DIR = os.path.join(PERSPECTIVE_DIR, "images")
    PERSPECTIVE_QUADS_DIR = os.path.join(PERSPECTIVE_DIR, "quads")
    PERSPECTIVE_MATRICES_DIR = os.path.join(PERSPECTIVE_DIR, "matrices")

    # Output: overwrite perspective_dataset with card-overlaid versions
    # We save to a sub-structure so the matrices/quads remain valid
    OUTPUT_DIR = "perspective_dataset_with_cards"
    OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
    OUTPUT_MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices")
    OUTPUT_QUADS_DIR = os.path.join(OUTPUT_DIR, "quads")
    OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

    for d in [OUTPUT_IMAGES_DIR, OUTPUT_MATRICES_DIR, OUTPUT_QUADS_DIR, OUTPUT_LABELS_DIR]:
        os.makedirs(d, exist_ok=True)

    # --- Load synthetic cards ---
    card_files = sorted(glob.glob(os.path.join(SYNTHETIC_CARDS_DIR, "*.png")))
    if not card_files:
        print("Error: No synthetic cards found in dataset/images/")
        return
    print(f"Found {len(card_files)} synthetic cards.")

    # --- Load perspective samples ---
    perspective_files = sorted(glob.glob(os.path.join(PERSPECTIVE_IMAGES_DIR, "*.jpg")))
    if not perspective_files:
        print("Error: No perspective images found in perspective_dataset/images/")
        return
    print(f"Found {len(perspective_files)} perspective backgrounds.")

    # --- Generate 1200 samples ---
    NUM_SAMPLES = 1200
    print(f"\nGenerating {NUM_SAMPLES} overlay samples...\n")

    success_count = 0
    for i in range(NUM_SAMPLES):
        # Pick the i-th synthetic card (cycle if needed)
        card_path = card_files[i % len(card_files)]
        card_basename = os.path.splitext(os.path.basename(card_path))[0]

        # Pick a random perspective background
        bg_path = random.choice(perspective_files)
        bg_basename = os.path.splitext(os.path.basename(bg_path))[0]

        # Load quad for this background
        quad_path = os.path.join(PERSPECTIVE_QUADS_DIR, f"{bg_basename}.json")
        matrix_path = os.path.join(PERSPECTIVE_MATRICES_DIR, f"{bg_basename}.npy")

        if not os.path.exists(quad_path) or not os.path.exists(matrix_path):
            continue

        # Load images
        bg_img = cv2.imread(bg_path)
        card_img = cv2.imread(card_path)

        if bg_img is None or card_img is None:
            continue

        # Load quad coordinates
        with open(quad_path, 'r') as f:
            quad = np.array(json.load(f)["quad"], dtype="float32")

        # Overlay the synthetic card onto the background
        result_img = overlay_card(bg_img, card_img, quad)

        # --- Save results ---
        output_name = f"sample_{i:04d}"

        # Save image
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{output_name}.jpg"), result_img)

        # Copy the perspective matrix (still valid - same quad region)
        shutil.copy(matrix_path, os.path.join(OUTPUT_MATRICES_DIR, f"{output_name}.npy"))

        # Copy the quad (still valid - same positions)
        with open(os.path.join(OUTPUT_QUADS_DIR, f"{output_name}.json"), 'w') as f:
            json.dump({"quad": quad.tolist()}, f)

        # Copy the card label
        label_path = os.path.join(SYNTHETIC_LABELS_DIR, f"{card_basename}.txt")
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUTPUT_LABELS_DIR, f"{output_name}.txt"))
        else:
            with open(os.path.join(OUTPUT_LABELS_DIR, f"{output_name}.txt"), "w") as f:
                f.write("unknown")

        success_count += 1

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{NUM_SAMPLES}")

    print(f"\nDone! Generated {success_count} images.")
    print(f"Saved to: {OUTPUT_DIR}/")
    print(f"  - images/   : Perspective backgrounds with synthetic cards overlaid")
    print(f"  - matrices/ : Perspective straighten matrices (unchanged)")
    print(f"  - quads/    : Quad coordinates (unchanged)")
    print(f"  - labels/   : Card number labels from synthetic cards")


if __name__ == "__main__":
    main()
