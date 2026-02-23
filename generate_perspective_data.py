import cv2
import numpy as np
import json
import random
from pathlib import Path

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    
    return rect

def random_affine_matrix(img_w, img_h):
    # Generate random affine matrix (rotation, scale, translation)
    cx, cy = img_w / 2, img_h / 2
    angle, scale = random.uniform(-30, 30), random.uniform(0.5, 1.2)
    tx, ty = random.uniform(-0.2, 0.2) * img_w, random.uniform(-0.2, 0.2) * img_h

    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty
    
    return M_rot

def transform_quad(quad, M_affine):
    # Apply affine transformation to the 4 points
    quad_reshaped = np.array([quad], dtype="float32")
    return cv2.transform(quad_reshaped, M_affine)[0]

def is_quad_in_bounds(quad, img_w, img_h):
    # Ensure all points remain inside the image frame
    x_out = np.any((quad[:, 0] < 0) | (quad[:, 0] >= img_w))
    y_out = np.any((quad[:, 1] < 0) | (quad[:, 1] >= img_h))
    return not (x_out or y_out)

def compute_straighten_matrix(quad, output_size=(512, 323)):
    # Compute 3x3 perspective matrix to straighten the card
    out_w, out_h = output_size
    src_pts = order_points(quad)
    dst_pts = np.array([
        [0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]
    ], dtype="float32")
    
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def load_source_images(base_img_dir, base_json_dir):
    # Load all valid image paths and their quad coordinates
    img_path_obj, json_path_obj = Path(base_img_dir), Path(base_json_dir)
    sources = []

    for sub in ["DG", "DX", "LG", "LX"]:
        sub_img_dir, sub_json_dir = img_path_obj / sub, json_path_obj / sub
        if not sub_img_dir.exists(): continue
            
        for img_file in sub_img_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                json_file = sub_json_dir / f"{img_file.stem}.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        quad = np.array(json.load(f)["quad"], dtype="float32")
                    sources.append((str(img_file), quad))
                    
    return sources

def generate_dataset(num_samples=50, output_dir="perspective_dataset"):
    out_dir = Path(output_dir)
    for d in ["images", "matrices", "quads"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
        
    sources = load_source_images("temp_extract/images", "temp_extract/ground_truth")
    if not sources: return

    for i in range(num_samples):
        img_path, quad = random.choice(sources)
        img = cv2.imread(img_path)
        if img is None: continue
            
        h, w = img.shape[:2]
        valid_transform, attempts = False, 0
        
        # Find a valid transformation without clipping the card
        while not valid_transform and attempts < 10:
            M_affine = random_affine_matrix(w, h)
            new_quad = transform_quad(quad, M_affine)
            valid_transform = is_quad_in_bounds(new_quad, w, h)
            attempts += 1
        
        if not valid_transform: continue

        # Apply transform and compute Ground Truth
        transformed_img = cv2.warpAffine(img, M_affine, (w, h))
        straighten_mat = compute_straighten_matrix(new_quad)
        
        # Save results
        name = f"sample_{i:04d}"
        cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), transformed_img)
        np.save(str(out_dir / "matrices" / f"{name}.npy"), straighten_mat)
        with open(out_dir / "quads" / f"{name}.json", 'w') as f:
            json.dump({"quad": order_points(new_quad).tolist()}, f)

if __name__ == "__main__":
    generate_dataset(num_samples=50)