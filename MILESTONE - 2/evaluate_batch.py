import os
import cv2
import pandas as pd
from module_4 import run_ultimate_inference, load_model # Import your existing functions

# --- CONFIGURATION ---
TEST_DATA_DIR = r"D:\PCB\PCB_DATASET\images"
CATEGORIES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
REPORT_OUTPUT = r"D:\PCB\module4_output\final_metrics_report.csv"

def evaluate_system():
    model = load_model()
    results = []
    
    # Metrics counters
    stats = {cat: {"TP": 0, "FP": 0, "FN": 0, "Total": 0} for cat in CATEGORIES}

    print(" Starting Batch Evaluation...")

    for category in CATEGORIES:
        cat_path = os.path.join(TEST_DATA_DIR, category)
        if not os.path.exists(cat_path): continue
        
        # Test a subset (e.g., first 10 images) from each category for the report
        test_files = os.listdir(cat_path)[:10] 
        
        for filename in test_files:
            img_path = os.path.join(cat_path, filename)
            inference_res = run_ultimate_inference(img_path, model)
            
            stats[category]["Total"] += 1
            
            if inference_res is None or len(inference_res["evidence"]) == 0:
                # System missed the defect entirely
                stats[category]["FN"] += 1
                continue

            # Check each detection in the image
            found_correct_label = False
            for det in inference_res["evidence"]:
                if det["label"] == category:
                    stats[category]["TP"] += 1
                    found_correct_label = True
                else:
                    # System flagged something else or misclassified
                    stats[category]["FP"] += 1
            
            # If the image was processed but the specific correct defect wasn't among results
            if not found_correct_label:
                stats[category]["FN"] += 1

    # --- 3. GENERATE FINAL TABLE ---
    report_data = []
    for cat, val in stats.items():
        accuracy = (val["TP"] / val["Total"]) * 100 if val["Total"] > 0 else 0
        report_data.append({
            "Defect Type": cat,
            "True Positives (TP)": val["TP"],
            "False Positives (FP)": val["FP"],
            "False Negatives (FN)": val["FN"],
            "Success Rate (%)": f"{accuracy:.2f}%"
        })

    df = pd.DataFrame(report_data)
    df.to_csv(REPORT_OUTPUT, index=False)
    print(f" Evaluation Complete! Report saved to: {REPORT_OUTPUT}")
    print(df)

if __name__ == "__main__":
    evaluate_system()