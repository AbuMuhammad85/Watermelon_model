import os
import random
from PIL import Image

def get_sample_images():
    """
    Scans the local dataset directories to retrieve actual healthy, diseased,
    and non-watermelon leaf images to use as one-click test samples.
    """
    samples = {
        "healthy": None,
        "diseased": None,
        "non_watermelon": None
    }
    
    # 1. Healthy Watermelon
    healthy_dir = "data/processed/test/watermelon___healthy"
    if os.path.exists(healthy_dir):
        files = [f for f in os.listdir(healthy_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        if files:
            samples["healthy"] = os.path.join(healthy_dir, files[0])
            
    # 2. Diseased Watermelon (try mosaic_virus or anthracnose)
    diseased_dir = "data/processed/test/watermelon___mosaic_virus"
    if not os.path.exists(diseased_dir) or not os.listdir(diseased_dir):
        diseased_dir = "data/processed/test/watermelon___anthracnose"
        
    if os.path.exists(diseased_dir):
        files = [f for f in os.listdir(diseased_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        if files:
            samples["diseased"] = os.path.join(diseased_dir, files[0])
            
    # 3. Non-Watermelon (negatives from detector test set)
    non_wm_dir = "data/detector/test/not_watermelon"
    if os.path.exists(non_wm_dir):
        files = [f for f in os.listdir(non_wm_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        if files:
            samples["non_watermelon"] = os.path.join(non_wm_dir, files[0])
            
    return samples

def save_uploaded_file(uploaded_file, temp_dir="reports/temp"):
    """
    Saves an uploaded Streamlit file to a temporary location on disk so that
    it can be processed by tf.io.read_file (which requires disk paths).
    """
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path
