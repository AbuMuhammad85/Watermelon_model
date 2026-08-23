import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Custom import modules from app/
from app.inference.model_loader import load_keras_model, load_tflite_interpreter
from app.inference.preprocessing import preprocess_image
from app.inference.predictor import predict_crop_and_disease
from app.inference.tflite_inference import run_tflite_inference
from app.explainability.gradcam import run_gradcam_explanation
from app.utils.labels import get_label_info, DISEASE_LABELS
from app.utils.image_utils import get_sample_images, save_uploaded_file
from app.utils.metrics import load_evaluation_metrics

# Set Page Config
st.set_page_config(
    page_title="Noma AI — Model Validation Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Watermelon-themed CSS
st.markdown("""
    <style>
    .main {
        background-color: #FBFDFB;
    }
    .stApp {
        color: #2E4A3F;
    }
    h1, h2, h3 {
        color: #1B4332;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .stButton>button {
        background-color: #2D6A4F;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B4332;
        color: #D8F3DC;
        border: none;
    }
    /* Metric Cards styling */
    div[data-testid="metric-container"] {
        background-color: #F4F9F4;
        border: 1px solid #D8F3DC;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricValue"] {
        color: #2D6A4F;
        font-weight: 700;
    }
    /* Accent containers */
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 6px solid;
    }
    .card-confident {
        background-color: #E8F5E9;
        border-left-color: #2E7D32;
        color: #1B5E20;
    }
    .card-uncertain {
        background-color: #FFF3E0;
        border-left-color: #EF6C00;
        color: #E65100;
    }
    .card-rejected {
        background-color: #FFEBEE;
        border-left-color: #C62828;
        color: #B71C1C;
    }
    .result-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .result-subtitle {
        font-size: 16px;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITS -----------------
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'active_image_path' not in st.session_state:
    st.session_state.active_image_path = None
if 'active_image_label' not in st.session_state:
    st.session_state.active_image_label = None

# Scan local directories for sample images
sample_images = get_sample_images()

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("# ⚙️ Model Configuration")

# Format selection
model_format_option = st.sidebar.selectbox(
    "Select Model Format",
    options=["TensorFlow / Keras (H5)", "TFLite Float32", "TFLite Float16", "TFLite INT8"],
    index=1  # Use Float32 TFLite as the numerical reference for inference
)

# Set model paths based on selection
model_mapping = {
    "TensorFlow / Keras (H5)": {
        "format": "keras",
        "detector": "models/detector/best_detector_model.h5",
        "classifier": "models/disease/best_disease_model.h5",
        "precision": "Float32"
    },
    "TFLite Float32": {
        "format": "tflite",
        "detector": "exports/detector/watermelon_detector_float32.tflite",
        "classifier": "exports/disease/watermelon_disease_float32.tflite",
        "precision": "Float32"
    },
    "TFLite Float16": {
        "format": "tflite",
        "detector": "exports/detector/watermelon_detector_float16.tflite",
        "classifier": "exports/disease/watermelon_disease_float16.tflite",
        "precision": "Float16"
    },
    "TFLite INT8": {
        "format": "tflite",
        "detector": "exports/detector/watermelon_detector_int8.tflite",
        "classifier": "exports/disease/watermelon_disease_int8.tflite",
        "precision": "Integer (INT8)"
    }
}

active_config = model_mapping[model_format_option]

# Get model sizes dynamically
def get_file_size_mb(path):
    if os.path.exists(path):
        return f"{os.path.getsize(path) / (1024 * 1024):.2f} MB"
    return "N/A"

detector_size = get_file_size_mb(active_config["detector"])
classifier_size = get_file_size_mb(active_config["classifier"])

# Threshold Sliders
st.sidebar.markdown("### 🎚️ Decision Thresholds")
detector_threshold = st.sidebar.slider(
    "Watermelon Detector Threshold (T_det)",
    min_value=0.1, max_value=0.99, value=0.50, step=0.01,
    help="Minimum confidence required to classify an image as a watermelon leaf."
)
disease_threshold = st.sidebar.slider(
    "Disease Classifier Threshold (T_dis)",
    min_value=0.1, max_value=0.99, value=0.70, step=0.01,
    help="Minimum confidence required to diagnose a specific leaf disease confidently."
)

# Display model metadata in sidebar
st.sidebar.markdown("### ℹ️ Active Model Metadata")
st.sidebar.info(f"""
* **Backbone Architecture**: MobileNetV2
* **Input Image Size**: 224 × 224 × 3
* **Number of Disease Classes**: 4
* **Quantization Precision**: {active_config['precision']}
* **Detector File Size**: {detector_size}
* **Classifier File Size**: {classifier_size}
* **Inference Host runtime**: CPU (TensorFlow Native)
""")

# Load models based on selected format
detector_model = None
disease_model = None

if active_config["format"] == "keras":
    detector_model = load_keras_model(active_config["detector"])
    disease_model = load_keras_model(active_config["classifier"])
else:
    detector_model = load_tflite_interpreter(active_config["detector"])
    disease_model = load_tflite_interpreter(active_config["classifier"])

# ----------------- MAIN HEADER -----------------
st.title("🌿 NOMA AI — Model Validation Dashboard")
st.subheader("Standalone ML Testing & Evaluation Environment")
st.markdown("""
*This dashboard evaluates the two-stage cascading vision pipeline designed for offline smartphone leaf diagnostics.*
*Upload a crop image below to evaluate binary domain rejection, disease classification probability distributions, edge latency, and Grad-CAM explainability.*
""")

# Setup tab navigation
tabs = st.tabs(["📷 Upload & Analyze", "🧪 Live Test Cases", "📂 Batch Testing", "📊 Evaluation Reports", "ℹ️ System Info"])

# ----------------- TAB 1: UPLOAD & ANALYZE -----------------
with tabs[0]:
    st.header("📷 Upload an Image")
    col1, col2 = st.columns([1, 1])
    
    uploaded_file = None
    camera_file = None
    
    with col1:
        uploaded_file = st.file_uploader("Choose a leaf image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])
        
        # Camera input support
        with st.expander("Or Capture Image using Device Camera"):
            camera_file = st.camera_input("Take a photo of a watermelon leaf")
            
        active_image = uploaded_file or camera_file
        
        if active_image is not None:
            # Save uploaded image locally to pass to Grad-CAM and preprocessing
            st.session_state.active_image_path = save_uploaded_file(active_image)
            st.session_state.active_image_label = active_image.name
            
            # Reset analysis state if a new file is uploaded
            st.session_state.analysis_run = False
            
            # Show original image metadata
            img = Image.open(st.session_state.active_image_path)
            st.image(img, caption="Uploaded Original Image", use_container_width=True)
            
            st.write(f"**Image Dimensions**: {img.width}x{img.height} pixels | **Format**: {img.format} | **Size**: {os.path.getsize(st.session_state.active_image_path)/1024:.1f} KB")
            
            # Trigger analysis
            if st.button("🔍 Analyze Image", type="primary"):
                st.session_state.analysis_run = True

    # ----------------- Inference execution -----------------
    with col2:
        if st.session_state.active_image_path is not None and st.session_state.analysis_run:
            if detector_model is None or disease_model is None:
                st.error("Error: Models are not loaded. Check model paths in configuration.")
            else:
                with st.spinner("Processing analysis pipeline..."):
                    # Safe prediction execution handling both dict and tuple returns
                    prediction_output = predict_crop_and_disease(
                        image=st.session_state.active_image_path,
                        detector_model=detector_model,
                        disease_model=disease_model,
                        model_format=active_config["format"],
                        detector_threshold=detector_threshold,
                        disease_threshold=disease_threshold
                    )
                    
                    if isinstance(prediction_output, tuple):
                        result = prediction_output[0]
                    else:
                        result = prediction_output
                    
                # ----------------- Display results card -----------------
                st.subheader("🏁 Classification Output")
                
                status = result["status"]
                
                # CONFIDENT STATE
                if status == "confident":
                    hausa_desc = get_label_info(result["disease"], "ha")
                    st.markdown(f"""
                    <div class="result-card card-confident">
                        <div class="result-title">🍉 Watermelon Detected (Ganyen Kankana)</div>
                        <div class="result-subtitle"><strong>Diagnosis:</strong> {get_label_info(result['disease'], 'en')}</div>
                        <div class="result-subtitle"><strong>Karin Bayani (Hausa):</strong> {hausa_desc}</div>
                        <div class="result-subtitle"><strong>Pipeline Status:</strong> {get_label_info('confident', 'ha')} (CONFIDENT)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # UNCERTAIN STATE
                elif status == "uncertain":
                    st.markdown(f"""
                    <div class="result-card card-uncertain">
                        <div class="result-title">⚠️ Uncertain Diagnosis (Babu Cikakken Tabbas)</div>
                        <div class="result-subtitle">The model identified a watermelon leaf but is not confident in the specific disease class.</div>
                        <div class="result-subtitle"><strong>Hausa Guideline:</strong> {get_label_info('uncertain', 'ha')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("⚠️ Recommendation: Adjust photo alignment or provide a clearer image with less glare/background clutter.")
                    
                # REJECTED STATE
                elif status == "not_watermelon":
                    st.markdown(f"""
                    <div class="result-card card-rejected">
                        <div class="result-title">❌ Image Classified as NOT WATERMELON</div>
                        <div class="result-subtitle">The detector confidence for watermelon (<strong>{result['watermelon_confidence']:.2%}</strong>) is below the configured threshold (<strong>{detector_threshold:.2%}</strong>).</div>
                        <div class="result-subtitle">The image may contain another crop, background vegetation, or an outdoor field condition that the current baseline detector does not recognize as a watermelon leaf.</div>
                        <div class="result-subtitle"><strong>Hausa translation:</strong> {get_label_info('not_watermelon', 'ha')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.error("❌ The disease classification step was bypassed for safety. No disease diagnosis is displayed for rejected crops.")

                # Latency & performance stats
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Inference Latency", f"{result['total_latency_ms']:.1f} ms")
                with m_col2:
                    confidence_label = "Watermelon Confidence" if result["is_watermelon"] else "Non-Watermelon Probability"
                    confidence_val = result["watermelon_confidence"] if result["is_watermelon"] else result.get("not_watermelon_probability", 1.0 - result["watermelon_confidence"])
                    st.metric(confidence_label, f"{confidence_val:.2%}")
                    
                st.progress(confidence_val)

                # Disease Probability Rank List
                if result["is_watermelon"]:
                    st.markdown("### 📊 Disease Probabilities")
                    
                    prob_df = pd.DataFrame([
                        {
                            "Disease": get_label_info(k, "en"),
                            "Hausa (Karin Bayani)": get_label_info(k, "ha"),
                            "Probability": f"{v:.2%}",
                            "raw_prob": v
                        } for k, v in result["disease_probabilities"].items()
                    ])
                    
                    st.table(prob_df[["Disease", "Hausa (Karin Bayani)", "Probability"]])
                    
                    if status == "confident":
                        st.success(f"✔️ Confidence score of **{result['disease_confidence']:.2%}** exceeds the safety threshold of **{disease_threshold:.2%}**.")
                    else:
                        st.info(f"ℹ️ Disease confidence of **{result['disease_confidence']:.2%}** falls below safety threshold of **{disease_threshold:.2%}**.")

                # ----------------- Technical Details Expander -----------------
                with st.expander("🔬 Technical Inference Details", expanded=False):
                    st.markdown("#### Detector Stage Details")
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**Model File**: `{active_config['detector']}`")
                        st.write(f"**Model Format**: `{active_config['format'].upper()}` ({active_config['precision']})")
                        st.write(f"**Input Tensor Shape**: `(1, 224, 224, 3)`")
                        st.write(f"**Normalization**: `mobilenet_v2.preprocess_input` ([-1.0, 1.0])")
                    with d2:
                        st.write(f"**Raw Sigmoid Output**: `{result.get('raw_detector_output', 0.0):.6f}`")
                        st.write(f"**Watermelon Probability**: `{result.get('watermelon_probability', 0.0):.4%}`")
                        st.write(f"**Non-Watermelon Probability**: `{result.get('not_watermelon_probability', 0.0):.4%}`")
                        st.write(f"**Detector Threshold ($T_{{det}}$)**: `{detector_threshold:.2f}`")
                        st.write(f"**Detector Decision**: `{result.get('detector_decision', 'N/A')}`")

                    st.markdown("---")
                    st.markdown("#### Disease Classifier Stage Details")
                    if result["is_watermelon"]:
                        st.write(f"**Classifier Status**: `EXECUTED`")
                        st.write(f"**Classifier Model**: `{active_config['classifier']}`")
                        st.write(f"**Disease Threshold ($T_{{dis}}$)**: `{disease_threshold:.2f}`")
                        st.write(f"**Top Predicted Class**: `{result.get('disease')}`")
                        st.write(f"**Top Class Confidence**: `{result.get('disease_confidence', 0.0):.4%}`")
                    else:
                        st.write(f"**Classifier Status**: `NOT RUN` (Bypassed by detector)")
                        st.write(f"**Reason**: Detector confidence ({result['watermelon_confidence']:.2%}) < Threshold ({detector_threshold:.2%})")

                    st.markdown("---")
                    st.markdown("#### Latency Breakdown")
                    st.write(f"**Detector Latency**: `{result['detector_latency_ms']:.1f} ms`")
                    st.write(f"**Classifier Latency**: `{result['classifier_latency_ms']:.1f} ms`")
                    st.write(f"**Total Inference Latency**: `{result['total_latency_ms']:.1f} ms`")

                # ----------------- Grad-CAM explainability -----------------
                st.markdown("---")
                st.subheader("🔥 Model Explanation (Grad-CAM)")
                
                if active_config["format"] == "keras":
                    with st.spinner("Generating Grad-CAM features..."):
                        gradcam_res = run_gradcam_explanation(
                            model=disease_model,
                            img_path=st.session_state.active_image_path,
                            model_format="keras"
                        )
                        
                    if gradcam_res is not None:
                        g_tabs = st.tabs(["Original", "Heatmap", "Overlay"])
                        with g_tabs[0]:
                            st.image(gradcam_res["original"], caption="Normalized Input Image (224x224)", use_container_width=True)
                        with g_tabs[1]:
                            st.image(gradcam_res["heatmap"], caption="Grad-CAM Activation Heatmap", use_container_width=True)
                        with g_tabs[2]:
                            st.image(gradcam_res["overlay"], caption="Superimposed Grad-CAM Map (Visual Focus Area)", use_container_width=True)
                            st.write("*Note: Red regions highlight areas that contributed most strongly to the model's disease prediction.*")
                    else:
                        st.warning("⚠️ Grad-CAM model tracing failed. Make sure the backbone contains the target convolutional layers.")
                else:
                    st.info("ℹ️ **Grad-CAM Explainability is available for TensorFlow / Keras H5 models.**\n\nThe currently active TFLite model is a flatbuffer binary optimized for edge execution and does not expose symbolic gradient tapes required for Grad-CAM.\n\nTo inspect Grad-CAM heatmaps, select **'TensorFlow / Keras (H5)'** in the sidebar configuration.")
        else:
            st.info("👈 Upload an image or capture a photo in the upload section to run model diagnostics.")

# ----------------- TAB 2: LIVE TEST CASES -----------------
with tabs[1]:
    st.header("🧪 One-Click Sample Testing")
    st.markdown("Click one of the buttons below to load prepared test images from the split datasets and verify rejection or disease predictions:")
    
    t_col1, t_col2, t_col3 = st.columns(3)
    
    sample_to_load = None
    
    with t_col1:
        if st.button("🥬 Test 1: Healthy Watermelon"):
            sample_to_load = sample_images.get("healthy")
            
    with t_col2:
        if st.button("🍂 Test 2: Diseased Watermelon (Mosaic)"):
            sample_to_load = sample_images.get("diseased")
            
    with t_col3:
        if st.button("🌶️ Test 3: Non-Watermelon Leaf (Negative)"):
            sample_to_load = sample_images.get("non_watermelon")
            
    if sample_to_load is not None:
        if os.path.exists(sample_to_load):
            st.session_state.active_image_path = sample_to_load
            st.session_state.active_image_label = os.path.basename(sample_to_load)
            st.session_state.analysis_run = True
            st.success(f"Loaded sample image: `{os.path.basename(sample_to_load)}`. Please switch back to the **📷 Upload & Analyze** tab to view classification reports, metrics, and Grad-CAM maps.")
        else:
            st.error(f"Sample file not found at: `{sample_to_load}`. Verify dataset processed folders exist.")

# ----------------- TAB 3: BATCH TESTING -----------------
with tabs[2]:
    st.header("📂 Batch File Evaluation")
    st.markdown("Upload multiple images to run batch inference and generate evaluation reports:")
    
    batch_files = st.file_uploader("Upload multiple crop images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if batch_files:
        st.write(f"Found {len(batch_files)} images for batch processing.")
        
        if st.button("🚀 Run Batch Inference"):
            batch_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(batch_files):
                temp_path = save_uploaded_file(file, temp_dir="reports/temp_batch")
                
                # Predict safely
                batch_pred_output = predict_crop_and_disease(
                    image=temp_path,
                    detector_model=detector_model,
                    disease_model=disease_model,
                    model_format=active_config["format"],
                    detector_threshold=detector_threshold,
                    disease_threshold=disease_threshold
                )
                
                if isinstance(batch_pred_output, tuple):
                    res = batch_pred_output[0]
                else:
                    res = batch_pred_output
                
                batch_results.append({
                    "Filename": file.name,
                    "Watermelon Detected": "YES" if res["is_watermelon"] else "NO",
                    "Watermelon Confidence": f"{res['watermelon_confidence']:.2%}",
                    "Prediction Status": res["status"].upper(),
                    "Diagnosed Disease": get_label_info(res["disease"], "en") if res["disease"] else "N/A",
                    "Disease Confidence": f"{res['disease_confidence']:.2%}" if res["disease_confidence"] else "N/A",
                    "Latency (ms)": f"{res['total_latency_ms']:.1f}"
                })
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                progress = (idx + 1) / len(batch_files)
                progress_bar.progress(progress)
                status_text.text(f"Processed {idx + 1}/{len(batch_files)} images...")
                
            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df)
            
            accepted_count = sum(1 for r in batch_results if r["Watermelon Detected"] == "YES")
            rejected_count = len(batch_results) - accepted_count
            avg_latency = np.mean([float(r["Latency (ms)"]) for r in batch_results])
            
            st.markdown("### 📊 Batch Statistics")
            s_col1, s_col2, s_col3 = st.columns(3)
            s_col1.metric("Accepted (Watermelon)", f"{accepted_count} / {len(batch_results)}")
            s_col2.metric("Rejected (Non-Watermelon)", f"{rejected_count} / {len(batch_results)}")
            s_col3.metric("Avg Inference Latency", f"{avg_latency:.1f} ms")
            
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results as CSV",
                data=csv_data,
                file_name="batch_inference_results.csv",
                mime="text/csv"
            )

# ----------------- TAB 4: EVALUATION REPORTS -----------------
with tabs[3]:
    st.header("📊 Model Evaluation & Training Reports")
    
    eval_metrics = load_evaluation_metrics()
    
    if not eval_metrics["raw_report"]:
        st.warning("⚠️ Pre-calculated metrics are unavailable because reports/disease_classification_report.txt was not found.")
    else:
        st.subheader("🧠 Disease Classification Metrics (Test Split)")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Disease Classifier Accuracy", f"{eval_metrics['accuracy']:.2%}")
        m2.metric("Macro F1-Score", f"{eval_metrics['macro_f1']:.4f}")
        m3.metric("Weighted F1-Score", f"{eval_metrics['weighted_f1']:.4f}")
        
        st.markdown("#### Full Classification Report")
        st.code(eval_metrics["raw_report"])
        
        st.markdown("---")
        st.subheader("🌀 Confusion Matrices")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            cm_disease_path = "reports/figures/disease_cm.png"
            if os.path.exists(cm_disease_path):
                st.image(cm_disease_path, caption="Disease Classifier Confusion Matrix", use_container_width=True)
            else:
                st.info("Disease Confusion Matrix plot not found.")
                
        with c_col2:
            cm_det_path = "reports/figures/detector_cm.png"
            if os.path.exists(cm_det_path):
                st.image(cm_det_path, caption="Binary Detector Confusion Matrix", use_container_width=True)
            else:
                st.info("Detector Confusion Matrix plot not found.")
                
        st.markdown("---")
        st.subheader("📈 Training History Curves")
        st.markdown("The following charts display the training history logs on CPU (Stage A: head training vs Stage B: fine-tuning epochs):")
        
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            stage_b_acc = "reports/figures/disease_stage_b_accuracy.png"
            if os.path.exists(stage_b_acc):
                st.image(stage_b_acc, caption="Disease Classifier Fine-Tuning Accuracy", use_container_width=True)
        with h_col2:
            stage_b_loss = "reports/figures/disease_stage_b_loss.png"
            if os.path.exists(stage_b_loss):
                st.image(stage_b_loss, caption="Disease Classifier Fine-Tuning Loss", use_container_width=True)
                
        if eval_metrics["ood_report"]:
            st.markdown("---")
            st.subheader("🛡️ Held-Out Out-of-Domain (OOD) Rejection Performance")
            st.markdown(eval_metrics["ood_report"])

# ----------------- TAB 5: SYSTEM INFO -----------------
with tabs[4]:
    st.header("ℹ️ Model Architecture & Verification Details")
    st.markdown("This tab details the tensor shapes, quantization parameters, and file metrics for the active model:")
    
    st.subheader("🖥️ Hardware & Platform Info")
    st.code(f"""
TensorFlow Version:  {tf.__version__}
Eager Execution:     {tf.executing_eagerly()}
Available CPUs:      {os.cpu_count()} cores
GPU Support Available: No CUDA GPUs detected (falling back to CPU runtime)
""")
    
    st.markdown("---")
    st.subheader("🧪 TFLite Quantization Parameters (INT8 Verification)")
    
    if "int8" in active_config["detector"].lower() or "int8" in active_config["classifier"].lower():
        st.success("✔️ ACTIVE CONFIGURATION IS INT8 QUANTIZED. VERIFICATION STATS BELOW:")
        
        for name, path in [("Binary Detector", active_config["detector"]), ("Disease Classifier", active_config["classifier"])]:
            if os.path.exists(path):
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                
                st.markdown(f"#### {name}")
                st.write(f"**Model Path**: `{path}`")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.markdown("**Input Details**")
                    st.write(f"* **Type**: `{input_details['dtype']}`")
                    st.write(f"* **Shape**: `{input_details['shape']}`")
                    st.write(f"* **Scale**: `{input_details['quantization'][0]}`")
                    st.write(f"* **Zero Point**: `{input_details['quantization'][1]}`")
                with v_col2:
                    st.markdown("**Output Details**")
                    st.write(f"* **Type**: `{output_details['dtype']}`")
                    st.write(f"* **Shape**: `{output_details['shape']}`")
                    st.write(f"* **Scale**: `{output_details['quantization'][0]}`")
                    st.write(f"* **Zero Point**: `{output_details['quantization'][1]}`")
    else:
        st.info("💡 Select the TFLite INT8 model configuration in the sidebar to inspect integer quantization scale parameters, input zero points, and data types.")
        
    if eval_metrics["quantization_report"]:
        st.markdown("---")
        st.subheader("📦 Model Quantization & Size Summary Report")
        st.markdown(eval_metrics["quantization_report"])