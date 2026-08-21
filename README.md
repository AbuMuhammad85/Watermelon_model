# Watermelon AI — Offline Image Understanding & Disease Diagnostic System

This project contains a production-oriented, offline-capable model suite designed to identify watermelon leaves, reject out-of-domain leaf species or background noise, classify disease states, and explain predictions using Grad-CAM.

---

## 🚀 How to Run the Streamlit Dashboard

A fully functional Streamlit dashboard has been built for model testing, edge quantization verification, batch image testing, and explainability exploration.

### 1. Activate the Virtual Environment
```powershell
.venv\Scripts\activate
```

### 2. Launch the Streamlit Dashboard
```powershell
streamlit run streamlit_app.py
```
* **Dashboard URL**: Once launched, the dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🛠️ CLI Predict & Batch Evaluate

### 1. Single-Image CLI Prediction (with Grad-CAM Heatmap)
```powershell
python predict.py --image data/processed/test/watermelon___healthy/IMG_2881.jpg --gradcam
```
* Generates prediction outputs in structured JSON format and outputs Grad-CAM maps at `reports/figures/gradcam_output.png`.

### 2. Folder-Wide Batch Evaluation
```powershell
python evaluate.py --dataset data/processed/test/
```
* Scans all subdirectories, evaluates cascaded classification thresholds, writes results to `reports/batch_predictions.csv`, and outputs failure cases to `reports/misclassified_images.txt`.

---

## 🧪 Running Automated Unit Tests

To run the complete test suite verifying dataset loaders, model definitions, and TFLite execution:
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📂 Project Architecture

```
c:\Users\lenovo\Dr Falalu Ibrahim\Model
├── app/                        # Streamlit Dashboard Modules
│   ├── inference/              # Preprocessing, caching model loaders, and TFLite execution
│   ├── explainability/         # Keras Grad-CAM wrapper
│   └── utils/                  # Hausa translations, sample loading, and metrics parsers
├── src/                        # Core AI Pipeline Package
│   ├── data/                   # Train/Val/Test leakage-safe splits and download scripts
│   ├── models/                 # Model definitions (Functional API)
│   ├── training/               # Stage-wise training scripts
│   ├── evaluation/             # Confusion matrices, ROC, and Grad-CAM feature mappings
│   └── inference/              # Pipeline orchestrator
├── models/                     # Saved Keras H5 models
├── exports/                    # Exported TFLite models (FP32, FP16, INT8)
├── reports/                    # Metric summaries, CSV logs, and Matplotlib figures
├── tests/                      # Automated unit test suite
├── streamlit_app.py            # Streamlit dashboard entrypoint
├── predict.py                  # CLI single image prediction utility
├── evaluate.py                 # CLI batch folder evaluation utility
└── requirements.txt            # Python package requirements
```
