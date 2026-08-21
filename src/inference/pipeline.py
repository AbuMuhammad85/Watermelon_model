import os
import numpy as np
import tensorflow as tf
from PIL import Image

class WatermelonAnalyzer:
    def __init__(self, detector_path=None, disease_path=None, labels_path=None):
        # Default paths
        if detector_path is None:
            detector_path = "models/detector/best_detector_model.h5"
            if not os.path.exists(detector_path):
                detector_path = "models/detector/final_detector_model.h5"
                
        if disease_path is None:
            disease_path = "models/disease/best_disease_model.h5"
            if not os.path.exists(disease_path):
                disease_path = "models/disease/final_disease_model.h5"
                
        if labels_path is None:
            labels_path = "models/disease/disease_labels.txt"
            
        print(f"Initializing analyzer...")
        print(f"  Loading detector: {detector_path}")
        self.detector = tf.keras.models.load_model(detector_path)
        
        print(f"  Loading disease classifier: {disease_path}")
        self.disease_classifier = tf.keras.models.load_model(disease_path)
        
        # Load labels
        self.labels = []
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.labels = [line.strip() for line in f if line.strip()]
        else:
            self.labels = ["Anthracnose", "Downy Mildew", "Healthy", "Mosaic Virus"]
        print(f"  Loaded labels: {self.labels}")
        
        # Calibrated thresholds (starting values, adjusted after validation analysis)
        self.detector_threshold = 0.50
        self.disease_threshold = 0.70 # Require 70% confidence to predict a disease

    def preprocess_image(self, img_path):
        """
        Loads and preprocesses image for MobileNetV2.
        """
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        
        # MobileNetV2 preprocessing: scale to [-1, 1]
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_tensor = tf.expand_dims(img_preprocessed, axis=0)
        return img_tensor

    def analyze(self, img_path):
        """
        Runs the two-stage inference pipeline.
        Returns a structured dictionary matching the integration contract.
        """
        try:
            # 1. Preprocess image
            img_tensor = self.preprocess_image(img_path)
            
            # 2. Stage 1: Detector
            det_pred = self.detector.predict(img_tensor, verbose=0)[0][0]
            
            # Check if not watermelon
            if det_pred < self.detector_threshold:
                return {
                    "status": "not_watermelon",
                    "is_watermelon": False,
                    "watermelon_confidence": float(1.0 - det_pred),
                    "disease": None,
                    "disease_confidence": None
                }
            
            # 3. Stage 2: Disease Classifier
            dis_preds = self.disease_classifier.predict(img_tensor, verbose=0)[0]
            pred_class_idx = np.argmax(dis_preds)
            pred_class_prob = dis_preds[pred_class_idx]
            pred_class_name = self.labels[pred_class_idx]
            
            # Check confidence for uncertainty handling
            if pred_class_prob < self.disease_threshold:
                return {
                    "status": "uncertain",
                    "is_watermelon": True,
                    "watermelon_confidence": float(det_pred),
                    "disease": None,
                    "disease_confidence": float(pred_class_prob)
                }
            
            return {
                "status": "confident",
                "is_watermelon": True,
                "watermelon_confidence": float(det_pred),
                "disease": pred_class_name,
                "disease_confidence": float(pred_class_prob)
            }
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {
                "status": "model_error",
                "is_watermelon": False,
                "watermelon_confidence": 0.0,
                "disease": None,
                "disease_confidence": 0.0
            }
