import time
import numpy as np
from app.inference.preprocessing import preprocess_image
from app.inference.tflite_inference import run_tflite_inference

def predict_crop_and_disease(
    image,
    detector_model,
    disease_model,
    model_format="keras",
    detector_threshold=0.85,
    disease_threshold=0.70,
    labels=None
):
    """
    Executes the two-stage cascade prediction on an image.
    Supports Keras H5 models and TFLite (Float32, Float16, INT8) models.
    """
    if labels is None:
        labels = ["watermelon___anthracnose", "watermelon___downy_mildew", "watermelon___healthy", "watermelon___mosaic_virus"]

    # 1. Preprocess image
    img_tensor = preprocess_image(image)
    
    # 2. Stage 1: Detector
    start_time = time.time()
    
    if model_format == "keras":
        det_pred = detector_model.predict(img_tensor, verbose=0)[0][0]
    else:
        # TFLite
        det_output = run_tflite_inference(detector_model, img_tensor)
        det_pred = float(det_output[0][0])
        
    detector_latency_ms = (time.time() - start_time) * 1000
    
    # Check if outside watermelon domain
    if det_pred < detector_threshold:
        res = {
            "status": "not_watermelon",
            "is_watermelon": False,
            "watermelon_confidence": float(1.0 - det_pred),
            "disease": None,
            "disease_confidence": None,
            "disease_probabilities": {label: 0.0 for label in labels},
            "detector_latency_ms": detector_latency_ms,
            "classifier_latency_ms": 0.0,
            "total_latency_ms": detector_latency_ms
        }
        return res, detector_latency_ms
        
    # Stage 2: Classifier
    disease_start_time = time.time()
    
    if model_format == "keras":
        dis_preds = disease_model.predict(img_tensor, verbose=0)[0]
    else:
        # TFLite
        dis_output = run_tflite_inference(disease_model, img_tensor)
        dis_preds = dis_output[0]
        
    classifier_latency_ms = (time.time() - disease_start_time) * 1000
    total_latency_ms = detector_latency_ms + classifier_latency_ms
    
    pred_class_idx = np.argmax(dis_preds)
    pred_class_prob = float(dis_preds[pred_class_idx])
    pred_class_name = labels[pred_class_idx]
    
    # Generate probabilities dictionary
    disease_probabilities = {labels[i]: float(dis_preds[i]) for i in range(len(labels))}
    
    # Sort disease probabilities descending
    disease_probabilities_sorted = dict(sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True))
    
    # Check if prediction is below threshold
    if pred_class_prob < disease_threshold:
        res = {
            "status": "uncertain",
            "is_watermelon": True,
            "watermelon_confidence": float(det_pred),
            "disease": None,
            "disease_confidence": float(pred_class_prob),
            "disease_probabilities": disease_probabilities_sorted,
            "detector_latency_ms": detector_latency_ms,
            "classifier_latency_ms": classifier_latency_ms,
            "total_latency_ms": total_latency_ms
        }
        return res, total_latency_ms
        
    res = {
        "status": "confident",
        "is_watermelon": True,
        "watermelon_confidence": float(det_pred),
        "disease": pred_class_name,
        "disease_confidence": float(pred_class_prob),
        "disease_probabilities": disease_probabilities_sorted,
        "detector_latency_ms": detector_latency_ms,
        "classifier_latency_ms": classifier_latency_ms,
        "total_latency_ms": total_latency_ms
    }
    return res, total_latency_ms