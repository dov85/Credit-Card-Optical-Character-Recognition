import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import string

def generate_random_data():
    # Generate 4 groups of 4 digits
    card_num_groups = [str(random.randint(1000, 9999)) for _ in range(4)]
    full_card_num = " ".join(card_num_groups)
    
    # Generate random date
    month = f"{random.randint(1, 12):02}"
    year = random.randint(25, 35)
    date_str = f"{month}/{year}"
    
    # Generate random ID
    id_part1 = random.randint(100, 999)
    id_part2 = random.randint(1000000, 9999999)
    check_digit = random.randint(0, 9)
    id_str = f"{id_part1} - {id_part2}\\{check_digit}"
    
    # Generate random name
    first_name = ''.join(random.choices(string.ascii_uppercase, k=random.randint(4, 7)))
    last_name = ''.join(random.choices(string.ascii_uppercase, k=random.randint(4, 7)))
    name_str = f"{first_name} {last_name}"
    
    return card_num_groups, full_card_num, date_str, id_str, name_str

def create_synthetic_dataset(base_image_path, num_samples=1200):
    if not os.path.exists(base_image_path):
        print(f"Error: Base file '{base_image_path}' not found.")
        return

    # Create directories
    base_dir = "dataset"
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Configuration
    font_name = "arial.ttf"
    font_path = f"C:\\Windows\\Fonts\\{font_name}"
    
    size_main = 19
    size_date = 12
    size_small_id = 12
    size_name = 10
    
    bg_color = (0, 0, 0)
    text_color = (200, 200, 200)

    try:
        f_main = ImageFont.truetype(font_path, size_main)
        f_date = ImageFont.truetype(font_path, size_date)
        f_id = ImageFont.truetype(font_path, size_small_id)
        f_name = ImageFont.truetype(font_path, size_name)
    except OSError:
        print("Font not found, using default.")
        f_main = f_date = f_id = f_name = ImageFont.load_default()

    # Load base image once
    original_pil = Image.open(base_image_path).convert("RGBA")
    img_width = original_pil.width

    print(f"Starting generation of {num_samples} images...")

    for i in range(num_samples):
        # 1. Generate Data
        groups, full_num_text, date_text, id_text, name_text = generate_random_data()
        
        # 2. Draw on Image
        img = original_pil.copy()
        draw = ImageDraw.Draw(img)

        # Draw Main Number
        draw.rectangle([(37, 98), (img_width - 41, 120)], fill=bg_color)
        start_x = 38
        start_y = 96
        gap = 8
        current_x = start_x

        for group in groups:
            draw.text((current_x, start_y), group, font=f_main, fill=text_color)
            group_width = draw.textlength(group, font=f_main)
            current_x += group_width + gap

        # Draw Date
        draw.rectangle([(105, 130), (145, 142)], fill=bg_color)
        draw.text((106, 128), date_text, font=f_date, fill=text_color)

        # Draw ID and Name
        draw.rectangle([(35, 145), (150, 170)], fill=bg_color)
        draw.text((36, 145), id_text, font=f_id, fill=text_color)
        draw.text((36, 157), name_text, font=f_name, fill=text_color)

        # 3. Crop Logic (OpenCV)
        opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Crop
            final_img = opencv_img[y+3:y+h-1, x+4:x+w-4]
            
            # 4. Save Image
            filename = f"card_{i:04d}"
            save_img_path = os.path.join(img_dir, f"{filename}.png")
            cv2.imwrite(save_img_path, final_img)
            
            # 5. Save Label (Text file with the card number)
            save_lbl_path = os.path.join(lbl_dir, f"{filename}.txt")
            with open(save_lbl_path, "w") as f:
                f.write(full_num_text)
                
            if i % 100 == 0:
                print(f"Generated {i}/{num_samples}...")
        else:
            print(f"Skipping index {i}, crop failed.")

    print("Done! Check the 'dataset' folder.")

if __name__ == "__main__":
    create_synthetic_dataset("card.jpeg", num_samples=1200)