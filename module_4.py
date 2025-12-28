import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = r"D:\PCB\best_pcb_model_(2).pth"
TEMPLATE_DIR = r"D:\PCB\PCB_DATASET\PCB_USED"
OUTPUT_DIR = r"D:\PCB\module4_output"
CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD TRAINED MODEL ---
def load_model():
    print("🚀 Loading Model from:", MODEL_PATH)
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- 2. ROBUST TEMPLATE SELECTOR ---
def find_best_template(test_img):
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_temp_path = None
    
    # Standardize size for fast template matching (as per your Module 1 code)
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
            
    return best_temp_path, best_score

# --- 3. INTENSE SUBTRACTION LOGIC (from your Module 1 code) ---
def get_intense_mask(test_img, temp_img):
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE for local contrast (Best for minute defects)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_test = clahe.apply(gray_test)
    gray_temp = clahe.apply(gray_temp)

    # 2. Absolute Difference
    diff = cv2.absdiff(gray_temp, gray_test)

    # 3. Adaptive Thresholding (Detects faint variations)
    mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

    # 4. Noise Filtering
    mask = cv2.medianBlur(mask, 3)

    # 5. Refinement with soft global threshold
    _, certainty_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask, certainty_mask)
    
    return final_mask

# --- 4. ULTIMATE INFERENCE PIPELINE ---
def run_ultimate_inference(test_img_path, model):
    test_img = cv2.imread(test_img_path)
    if test_img is None: return None
    
    # A. Find the Template board
    temp_path, score = find_best_template(test_img)
    if score < 0.6: 
        print("❌ No matching template found. Score too low.")
        return None
    
    temp_img = cv2.imread(temp_path)
    temp_img = cv2.resize(temp_img, (test_img.shape[1], test_img.shape[0]))

    # B. Generate Intense Mask
    combined_mask = get_intense_mask(test_img, temp_img)

    # C. Prediction & Evidence Gathering
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = test_img.copy()
    evidence_list = []

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"🔍 Found {len(contours)} potential regions. Analyzing...")

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 15: continue # Ignore tiny noise
        
        x, y, w_b, h_b = cv2.boundingRect(cnt)
        
        # Crop ROI with padding for context (Module 2 Style)
        roi = test_img[max(0,y-25):y+h_b+25, max(0,x-25):x+w_b+25]
        
        if roi.size == 0: continue

        # CNN Classification (Module 3)
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = transform(roi_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(roi_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            label = CLASS_NAMES[pred.item()]

        # Annotation: Box + Label + Confidence
        cv2.rectangle(annotated, (x, y), (x+w_b, y+h_b), (0, 0, 255), 2)
        cv2.putText(annotated, f"{label} {conf.item()*100:.1f}%", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        evidence_list.append({
            "id": i, "label": label, "conf": conf.item(), "roi": roi
        })

    return {
        "annotated": annotated,
        "mask": combined_mask,
        "template": temp_img,
        "evidence": evidence_list
    }

# --- 5. EXECUTION ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pcb_model = load_model()
    
    # Example: Running a Short Circuit test image
    test_p = r"D:\PCB\PCB_DATASET\images\Short\04_short_04.jpg"
    
    result = run_ultimate_inference(test_p, pcb_model)
    
    if result:
        cv2.imwrite(f"{OUTPUT_DIR}/final_annotated.jpg", result["annotated"])
        cv2.imwrite(f"{OUTPUT_DIR}/subtraction_mask.jpg", result["mask"])
        cv2.imwrite(f"{OUTPUT_DIR}/matched_template.jpg", result["template"])
        
        print(f"✅ Processed successfully!")
        print(f"📦 Evidence saved for {len(result['evidence'])} defects.")
        for item in result['evidence']:
            print(f"   - Detected: {item['label']} ({item['conf']*100:.2f}%)")
    else:
        print("❌ Inference failed.")