"""import cv2
import numpy as np
import os

# --- CONFIGURATION ---
DATASET_PATH = r"D:\PCB\PCB_DATASET"
OUTPUT_PATH = r"D:\PCB\module1_output"
CATEGORIES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
TEMPLATE_DIR = os.path.join(DATASET_PATH, "PCB_USED")

def find_best_template(test_img):
  #Scans PCB_USED to find the template that matches the test layout
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_temp = None
    
    # We resize test to a standard size for fast matching
    test_small = cv2.resize(gray_test, (200, 200))

    for temp_name in os.listdir(TEMPLATE_DIR):
        temp_path = os.path.join(TEMPLATE_DIR, temp_name)
        temp_img = cv2.imread(temp_path, 0) # Load as grayscale
        if temp_img is None: continue
        
        temp_small = cv2.resize(temp_img, (200, 200))
        
        # Calculate Correlation Coefficient
        result = cv2.matchTemplate(test_small, temp_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_temp = temp_path
            
    return best_temp, best_score

def process_pcb_v3():
    print("🚀 Starting Module 1: Auto-Matching Pipeline...")
    
    for category in CATEGORIES:
        img_dir = os.path.join(DATASET_PATH, "images", category)
        if not os.path.exists(img_dir): continue
        
        print(f"Checking: {category}...")
        
        for filename in os.listdir(img_dir):
            test_img = cv2.imread(os.path.join(img_dir, filename))
            if test_img is None: continue

            # 1. AUTO-FIND THE CORRECT TEMPLATE
            temp_path, score = find_best_template(test_img)
            
            # If score < 0.5, the images are too different
            if score < 0.6: 
                print(f"⚠️ Skipping {filename}: No matching template found.")
                continue

            temp_img = cv2.imread(temp_path)
            temp_img = cv2.resize(temp_img, (test_img.shape[1], test_img.shape[0]))

            # 2. SIMPLE ALIGNMENT (No 3D warping)
            # Since these are from the same dataset, they should be mostly aligned.
            # We just do a tiny bit of shift correction.
            gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            gray_temp = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray_temp, gray_test)

            # 3. THRESHOLDING
            _, mask = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
            
            # Remove tiny noise
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # 4. SAVE OUTPUT
            out_name = f"{category}_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_PATH, "defect_masks", out_name), mask)
            
            # Visual check
            combined = np.hstack((cv2.resize(temp_img, (400, 400)), 
                                 cv2.resize(test_img, (400, 400)), 
                                 cv2.cvtColor(cv2.resize(mask, (400, 400)), cv2.COLOR_GRAY2BGR)))
            cv2.imwrite(os.path.join(OUTPUT_PATH, "sample_visuals", out_name), combined)

    print("✅ All matching templates found and processed!")

if __name__ == "__main__":
    process_pcb_v3()  """








import cv2
import numpy as np
import os

# --- CONFIGURATION ---
DATASET_PATH = r"D:\PCB\PCB_DATASET"
OUTPUT_PATH = r"D:\PCB\module1_output"
CATEGORIES = ["Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper", "Missing_hole"]
TEMPLATE_DIR = os.path.join(DATASET_PATH, "PCB_USED")

def find_best_template(test_img):
    """Automatically finds the matching template from the golden set."""
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_temp_path = None
    
    # Standardize size for fast template matching
    test_small = cv2.resize(gray_test, (256, 256))

    for temp_name in os.listdir(TEMPLATE_DIR):
        temp_path = os.path.join(TEMPLATE_DIR, temp_name)
        temp_img = cv2.imread(temp_path, 0)
        if temp_img is None: continue
        
        temp_small = cv2.resize(temp_img, (256, 256))
        result = cv2.matchTemplate(test_small, temp_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_temp_path = temp_path
            
    return best_temp_path

def get_intense_mask(test_img, temp_img):
    """Deep-scan subtraction using local contrast and adaptive thresholding."""
    # 1. Grayscale
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This makes tiny copper defects 'pop' by normalizing local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_test = clahe.apply(gray_test)
    gray_temp = clahe.apply(gray_temp)

    # 3. Absolute Difference
    diff = cv2.absdiff(gray_temp, gray_test)

    # 4. Adaptive Thresholding
    # Calculates a threshold for every 11x11 block to detect faint variations
    mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

    # 5. Noise Filtering
    # Median blur removes 'salt and pepper' noise without destroying small defect edges
    mask = cv2.medianBlur(mask, 3)

    # 6. Refinement: Use a soft global threshold to remove remaining shadows
    _, certainty_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask, certainty_mask)
    
    return diff, final_mask

def process_pcb_intense():
    print("🔥 Starting Intense Defect Extraction (No Detail Left Behind)...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(OUTPUT_PATH, "defect_masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "sample_visuals"), exist_ok=True)

    for category in CATEGORIES:
        img_dir = os.path.join(DATASET_PATH, "images", category)
        if not os.path.exists(img_dir): continue
        
        print(f"📦 Deep-scanning Category: {category}")
        
        for filename in os.listdir(img_dir):
            test_img = cv2.imread(os.path.join(img_dir, filename))
            if test_img is None: continue

            # Auto-Find Template
            temp_path = find_best_template(test_img)
            temp_img = cv2.imread(temp_path)
            temp_img = cv2.resize(temp_img, (test_img.shape[1], test_img.shape[0]))

            # Extraction
            diff, mask = get_intense_mask(test_img, temp_img)

            # Save Result
            out_name = f"{category}_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_PATH, "defect_masks", out_name), mask)
            
            # Create 3-panel Visualization: Template | Test | Intense Mask
            h, w = 450, 450
            viz = np.hstack((cv2.resize(temp_img, (w, h)), 
                             cv2.resize(test_img, (w, h)), 
                             cv2.cvtColor(cv2.resize(mask, (w, h)), cv2.COLOR_GRAY2BGR)))
            
            cv2.imwrite(os.path.join(OUTPUT_PATH, "sample_visuals", out_name), viz)

    print(" Intense Module 1 Complete. Check sample_visuals for filled-in defects.")

if __name__ == "__main__":
    process_pcb_intense()