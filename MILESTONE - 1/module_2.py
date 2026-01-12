import cv2
import numpy as np
import os

# --- CONFIGURATION ---
MASK_PATH = r"D:\PCB\module1_output\defect_masks"
IMAGE_PATH = r"D:\PCB\PCB_DATASET\images"
OUTPUT_ROI_PATH = r"D:\PCB\module2_output\cropped_defects"
VISUAL_PATH = r"D:\PCB\module2_output\contour_visuals"

# Categories based on your dataset
CATEGORIES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

def run_module2_extraction():
    print(" Starting Module 2: Contour Detection & ROI Extraction...")
    
    # Create output directories
    for cat in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_ROI_PATH, cat), exist_ok=True)
    os.makedirs(VISUAL_PATH, exist_ok=True)

    # Padding: We add 20 pixels around the defect for CNN context
    PADDING = 20

    for filename in os.listdir(MASK_PATH):
        # 1. Load the mask and the corresponding original test image
        mask = cv2.imread(os.path.join(MASK_PATH, filename), 0)
        
        # Extract category and original name from filename 
        # (format: category_filename.jpg)
        sorted_categories = sorted(CATEGORIES, key=lambda x: -len(x))
        category = next((cat for cat in sorted_categories if filename.startswith(cat + "_")), None)
        if not category:
            print(f" Cannot determine category for {filename}")
            continue
        
        # Locate the original defect image to crop from
        # Note: We need to strip our prefix to get the original filename
        orig_filename = filename.replace(f"{category}_", "")
        orig_img_path = os.path.join(IMAGE_PATH, category, orig_filename)
        orig_img = cv2.imread(orig_img_path)
        
        if orig_img is None or mask is None: continue

        # 2. CONTOUR DETECTION
        # RETR_EXTERNAL: Only detect the outer boundary of the defect
        # CHAIN_APPROX_SIMPLE: Compress horizontal, vertical, and diagonal segments
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        viz_img = orig_img.copy()
        roi_count = 0

        for i, cnt in enumerate(contours):
            # Filter out tiny noise (less than 5 pixels area)
            if cv2.contourArea(cnt) < 5:
                continue
            
            # 3. BOUNDING BOX EXTRACTION
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw on visualization (Green box)
            cv2.drawContours(viz_img, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 4. CROPPING WITH PADDING
            # Ensure coordinates stay within image boundaries
            y1, y2 = max(0, y - PADDING), min(orig_img.shape[0], y + h + PADDING)
            x1, x2 = max(0, x - PADDING), min(orig_img.shape[1], x + w + PADDING)
            
            roi_crop = orig_img[y1:y2, x1:x2]
            
            # 5. SAVE ROI
            roi_name = f"{category}_{orig_filename.split('.')[0]}_roi_{i}.jpg"
            save_path = os.path.join(OUTPUT_ROI_PATH, category, roi_name)
            cv2.imwrite(save_path, roi_crop)
            roi_count += 1

        # Save the visualization of detected contours
        cv2.imwrite(os.path.join(VISUAL_PATH, f"viz_{filename}"), viz_img)

    print(f" Module 2 Complete! ROIs extracted to: {OUTPUT_ROI_PATH}")

if __name__ == "__main__":
    run_module2_extraction()