import cv2
import numpy as np
import os
import json
import random
import glob
import shutil

# --- CONFIGURATION ---

# 1. Input Paths (Backgrounds and JSONs)
# Assumes structure: temp_extract/images/DG/DG01_01.jpg
# And: temp_extract/ground_truth/DG/DG01_01.json
BACKGROUND_BASE_DIR = "temp_extract/images"
JSON_BASE_DIR = "temp_extract/ground_truth"

# 2. Synthetic Cards Paths (The 1200 cards you generated)
SYNTHETIC_CARDS_DIR = "dataset/images"
SYNTHETIC_LABELS_DIR = "dataset/labels"

# 3. Output Paths
OUTPUT_DIR = "final_augmented_dataset"
OUTPUT_IMGS_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_LBLS_DIR = os.path.join(OUTPUT_DIR, "labels")

# 4. Settings
# How many synthetic cards to generate per background image
VARIATIONS_PER_BACKGROUND = 10 

def order_points(pts):
    """
    Sorts coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    Required for correct perspective transformation.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum, Bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, Bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def get_coordinates_from_json(json_path):
    """
    Extracts the 'quad' list from the JSON file.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Look for the 'quad' key as per your file structure
        if "quad" in data:
            return np.array(data["quad"], dtype="float32")
        else:
            return None
    except Exception as e:
        print(f"Error reading JSON {json_path}: {e}")
        return None

def overlay_card(bg_img, card_img, dest_coords):
    """
    Warps the synthetic card and pastes it onto the background using a mask.
    """
    # 1. Source Points (Flat card corners)
    (h, w) = card_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # 2. Destination Points (From JSON)
    dst_pts = order_points(dest_coords)

    # 3. Perspective Matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 4. Warp the card
    warped_card = cv2.warpPerspective(card_img, matrix, (bg_img.shape[1], bg_img.shape[0]))

    # 5. Create Mask for precise blending
    # Create a white rectangle of the same size as the card
    mask_src = np.ones((h, w), dtype="uint8") * 255
    # Warp the white rectangle to match the perspective
    warped_mask = cv2.warpPerspective(mask_src, matrix, (bg_img.shape[1], bg_img.shape[0]))

    # 6. Blending
    # Invert mask (get background area)
    mask_inv = cv2.bitwise_not(warped_mask)
    
    # Black out the card area in the background image
    bg_bg = cv2.bitwise_and(bg_img, bg_img, mask=mask_inv)
    
    # Take only the card area from the warped image
    card_fg = cv2.bitwise_and(warped_card, warped_card, mask=warped_mask)

    # Add both
    result = cv2.add(bg_bg, card_fg)
    return result

def main():
    # Create output directories
    if not os.path.exists(OUTPUT_IMGS_DIR):
        os.makedirs(OUTPUT_IMGS_DIR)
    if not os.path.exists(OUTPUT_LBLS_DIR):
        os.makedirs(OUTPUT_LBLS_DIR)

    # Load list of all synthetic cards
    synthetic_cards_list = glob.glob(os.path.join(SYNTHETIC_CARDS_DIR, "*.png"))
    
    if len(synthetic_cards_list) == 0:
        print("Error: No synthetic cards found in dataset/images.")
        return
        
    print(f"Loaded {len(synthetic_cards_list)} synthetic cards.")

    # List of subfolders to process
    subfolders = ['DG', 'DX', 'LG', 'LX']
    
    global_counter = 0

    for subfolder in subfolders:
        # Define paths for the current subfolder
        current_img_dir = os.path.join(BACKGROUND_BASE_DIR, subfolder)
        current_json_dir = os.path.join(JSON_BASE_DIR, subfolder)
        
        if not os.path.exists(current_img_dir):
            print(f"Skipping {subfolder}: Folder not found.")
            continue
            
        print(f"Processing folder: {subfolder}...")
        
        # Iterate over all images in the subfolder
        for img_filename in os.listdir(current_img_dir):
            if img_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                # Construct full paths
                bg_img_path = os.path.join(current_img_dir, img_filename)
                
                # Find matching JSON (Same name, but .json extension)
                # Example: DG01_01.jpg -> DG01_01.json
                json_filename = os.path.splitext(img_filename)[0] + ".json"
                json_path = os.path.join(current_json_dir, json_filename)
                
                # Check if JSON exists
                if not os.path.exists(json_path):
                    print(f"Skipping {img_filename}: JSON not found.")
                    continue
                
                # Load JSON coordinates
                coords = get_coordinates_from_json(json_path)
                if coords is None:
                    continue
                
                # Load the background image once
                bg_img_original = cv2.imread(bg_img_path)
                if bg_img_original is None:
                    continue

                # Generate variations
                for i in range(VARIATIONS_PER_BACKGROUND):
                    # 1. Select a random synthetic card
                    random_card_path = random.choice(synthetic_cards_list)
                    card_img = cv2.imread(random_card_path)
                    
                    if card_img is None:
                        continue
                        
                    # 2. Get the label (card number) for this card
                    card_basename = os.path.splitext(os.path.basename(random_card_path))[0]
                    original_label_path = os.path.join(SYNTHETIC_LABELS_DIR, card_basename + ".txt")
                    
                    # 3. Create the augmented image
                    final_img = overlay_card(bg_img_original, card_img, coords)
                    
                    if final_img is not None:
                        # 4. Save the result
                        # New name format: aug_{folder}_{original_name}_{count}.jpg
                        new_filename = f"aug_{subfolder}_{os.path.splitext(img_filename)[0]}_{i}.jpg"
                        save_img_path = os.path.join(OUTPUT_IMGS_DIR, new_filename)
                        
                        cv2.imwrite(save_img_path, final_img)
                        
                        # 5. Save the label
                        # We copy the text file content to the new label file
                        save_lbl_path = os.path.join(OUTPUT_LBLS_DIR, os.path.splitext(new_filename)[0] + ".txt")
                        
                        if os.path.exists(original_label_path):
                            shutil.copy(original_label_path, save_lbl_path)
                        else:
                            # Fallback if label missing
                            with open(save_lbl_path, "w") as f:
                                f.write("unknown")
                        
                        global_counter += 1
                        
        print(f"Finished folder {subfolder}.")

    print("-" * 50)
    print(f"Processing Complete.")
    print(f"Total images generated: {global_counter}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()