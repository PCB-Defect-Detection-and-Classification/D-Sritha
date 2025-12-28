import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pandas as pd
from io import BytesIO
import time

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="PCB Defect AI Inspector", layout="wide", page_icon="🛡️")

MODEL_PATH = "D:\\PCB\\best_pcb_model_(2).pth"
TEMPLATE_DIR = r"D:\PCB\PCB_DATASET\PCB_USED"
CLASS_NAMES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize session state for Module 7 performance tracking
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for Module 7 professional styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_pcb_model():
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- 2. THE INTENSE PROCESSING ENGINE (Optimized for Module 7) ---
def find_best_template(test_img):
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    best_score, best_temp_path = -1, None
    test_small = cv2.resize(gray_test, (256, 256))

    for temp_name in os.listdir(TEMPLATE_DIR):
        temp_path = os.path.join(TEMPLATE_DIR, temp_name)
        temp_img = cv2.imread(temp_path, 0)
        if temp_img is None: continue
        temp_small = cv2.resize(temp_img, (256, 256))
        
        res = cv2.matchTemplate(temp_small, test_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score, best_temp_path = max_val, temp_path
            
    return best_temp_path, best_score

def run_full_inspection(test_img, temp_path, model):
    start_time = time.time()
    temp_img = cv2.imread(temp_path)
    temp_img = cv2.resize(temp_img, (test_img.shape[1], test_img.shape[0]))
    
    # Module 1: Intense Logic (CLAHE + Adaptive)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    diff = cv2.absdiff(clahe.apply(gray_temp), clahe.apply(gray_test))
    mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    mask = cv2.medianBlur(mask, 3)
    _, cert = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    final_mask = cv2.bitwise_and(mask, cert)

    # Module 2 & 3: Extraction & AI Classification
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = test_img.copy()
    evidence_crops = []
    logs = []
    
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 15: continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = test_img[max(0,y-25):y+h+25, max(0,x-25):x+w+25]
        
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = preprocess(roi_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(roi_tensor)
            conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
            label = CLASS_NAMES[pred.item()]

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(annotated, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        evidence_crops.append(roi)
        logs.append({"ID": i+1, "Defect Type": label, "Confidence": f"{conf.item()*100:.2f}%", "Coordinates": f"X:{x}, Y:{y}"})

    process_time = time.time() - start_time
    return temp_img, final_mask, annotated, logs, evidence_crops, process_time

# --- 3. STREAMLIT FRONTEND (Integrated Module 7) ---
st.title("Industrial PCB AI Inspector v2.0")
st.subheader("Fulfilling Module 7: Evaluation, Performance, & Result Exporting")

# Tabs for separate views
tab1, tab2 = st.tabs([" Live Inspection", " Performance Analytics"])

with tab1:
    st.sidebar.header("Control Panel")
    uploaded_file = st.sidebar.file_uploader("Upload Test PCB Image", type=['jpg', 'png', 'jpeg'])
    confidence_threshold = st.sidebar.slider("Template Matching Sensitivity", 0.0, 1.0, 0.6)

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        user_img = cv2.imdecode(file_bytes, 1)
        st.sidebar.image(user_img, caption="Current Upload", use_container_width=True)

        if st.sidebar.button(" Start Full Inspection"):
            with st.spinner("Executing optimized pipeline..."):
                model = load_pcb_model()
                temp_path, score = find_best_template(user_img)
                
                if score < confidence_threshold:
                    st.error(f" Template Match Failed (Score: {score:.2f}). PCB design not recognized.")
                else:
                    ref_img, mask_img, final_img, logs, crops, p_time = run_full_inspection(user_img, temp_path, model)
                    
                    # Store in session state for Module 7 analytics
                    st.session_state.history.append({"match_score": score, "defects": len(logs), "time": p_time})

                    # Performance KPIs
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Template Match", f"{score*100:.1f}%")
                    m2.metric("Defects Found", len(logs))
                    m3.metric("Processing Time", f"{p_time:.3f} sec")

                    # Step-by-Step Evidence
                    st.markdown("###  Evidence Chain")
                    ev_col1, ev_col2, ev_col3 = st.columns(3)
                    ev_col1.image(ref_img, caption="Golden Reference")
                    ev_col2.image(mask_img, caption="Subtraction Mask")
                    ev_col3.image(final_img, caption="Final Prediction")

                    # ROI Gallery
                    if crops:
                        st.markdown("###  Defect Gallery")
                        cols = st.columns(min(len(crops), 5))
                        for i, crop in enumerate(crops):
                            with cols[i % 5]:
                                st.image(crop, caption=f"ID #{i+1}: {logs[i]['Defect Type']}")

                    # Export Section (Module 7 Task)
                    st.markdown("###  Export Inspection Data")
                    df = pd.DataFrame(logs)
                    st.dataframe(df, use_container_width=True)

                    c1, c2 = st.columns(2)
                    _, img_buffer = cv2.imencode(".jpg", final_img)
                    c1.download_button(" Save Annotated Image", data=BytesIO(img_buffer), file_name="output_labeled.jpg")
                    
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    c2.download_button(" Export CSV Log", data=csv_data, file_name="defect_report.csv")
    else:
        st.info("👋 Upload a PCB image to begin inspection.")

with tab2:
    st.header("System Evaluation & Optimization Metrics")
    if st.session_state.history:
        h_df = pd.DataFrame(st.session_state.history)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Match Score", f"{h_df['match_score'].mean()*100:.1f}%")
        c2.metric("Total Defects Flagged", int(h_df['defects'].sum()))
        c3.metric("Avg Latency", f"{h_df['time'].mean():.3f}s")
        
        st.line_chart(h_df['time'], use_container_width=True)
        st.markdown(" The system maintains low latency (<1s) by utilizing model caching and spatial downsampling.")
    else:
        st.warning("No data available. Run at least one inspection to view evaluation metrics.")

# Footer satisfying presentation requirements
st.markdown("---")
st.caption("Developed for Automated PCB Inspection System ")