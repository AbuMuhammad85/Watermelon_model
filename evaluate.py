import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.inference.pipeline import WatermelonAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Watermelon AI - Batch Evaluation Tool")
    parser.add_argument("--dataset", type=str, required=True, help="Path to test dataset directory")
    parser.add_argument("--output_csv", type=str, default="reports/batch_predictions.csv", help="Path to save predictions CSV")
    parser.add_argument("--output_misclassified", type=str, default="reports/misclassified_images.txt", help="Path to save misclassified images txt")
    return parser.parse_args()

def get_ground_truth(file_path):
    """
    Infers the true class from directory structure.
    Returns: (is_watermelon, disease_class_name or None)
    """
    parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
    
    # Positive classes
    if "anthracnose" in parent_dir:
        return True, "watermelon___anthracnose"
    elif "downy_mildew" in parent_dir:
        return True, "watermelon___downy_mildew"
    elif "healthy" in parent_dir:
        return True, "watermelon___healthy"
    elif "mosaic_virus" in parent_dir:
        return True, "watermelon___mosaic_virus"
    elif "watermelon" in parent_dir and "not" not in parent_dir:
        return True, "watermelon___healthy" # default positive
        
    # Negatives or OOD
    return False, None

def run_batch_evaluation():
    args = parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        return
        
    # Check if models exist
    detector_path = "models/detector/best_detector_model.h5"
    if not os.path.exists(detector_path):
        detector_path = "models/detector/final_detector_model.h5"
        
    disease_path = "models/disease/best_disease_model.h5"
    if not os.path.exists(disease_path):
        disease_path = "models/disease/final_disease_model.h5"
        
    if not os.path.exists(detector_path) or not os.path.exists(disease_path):
        print("Error: Models have not been trained yet. Please run training scripts first.")
        return
        
    print("Initializing Watermelon Analyzer...")
    analyzer = WatermelonAnalyzer()
    
    print(f"Scanning directory: {args.dataset}")
    image_files = []
    for root, _, files in os.walk(args.dataset):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
                
    print(f"Found {len(image_files)} images to evaluate.")
    if not image_files:
        return
        
    results = []
    misclassified = []
    
    correct_detector = 0
    correct_disease = 0
    disease_eligible = 0
    
    confidences = []
    
    for idx, path in enumerate(image_files):
        true_is_watermelon, true_disease = get_ground_truth(path)
        
        # Run prediction
        res = analyzer.analyze(path)
        
        pred_is_watermelon = res["is_watermelon"]
        pred_disease = res["disease"]
        status = res["status"]
        
        # Normalize predicted disease class name if present
        if pred_disease:
            if pred_disease.startswith("watermelon___"):
                pred_disease_normalized = pred_disease
            else:
                pred_disease_normalized = f"watermelon___{pred_disease.lower().replace(' ', '_')}"
        else:
            pred_disease_normalized = None
            
        # Check detector correctness
        detector_match = (pred_is_watermelon == true_is_watermelon)
        if detector_match:
            correct_detector += 1
            
        # Check disease correctness
        disease_match = False
        if true_is_watermelon:
            disease_eligible += 1
            if status == "confident" and pred_disease_normalized == true_disease:
                correct_disease += 1
                disease_match = True
            elif status == "uncertain":
                # Mark as uncertain error or handle separately
                pass
        
        # Log failure modes
        is_error = False
        error_type = ""
        
        if not detector_match:
            is_error = True
            error_type = "False Positive" if pred_is_watermelon else "False Negative"
        elif true_is_watermelon and not disease_match:
            is_error = True
            if status == "uncertain":
                error_type = "Uncertainty Reject (True Positive)"
            else:
                error_type = f"Misclassified Disease (Pred: {pred_disease}, True: {true_disease})"
                
        if is_error:
            misclassified.append({
                "path": path,
                "error_type": error_type,
                "status": status,
                "watermelon_conf": res["watermelon_confidence"],
                "disease_conf": res["disease_confidence"]
            })
            
        # Record results for CSV
        results.append({
            "file_path": path,
            "true_is_watermelon": true_is_watermelon,
            "true_disease": true_disease,
            "pred_is_watermelon": pred_is_watermelon,
            "pred_disease": pred_disease,
            "status": status,
            "watermelon_conf": res["watermelon_confidence"],
            "disease_conf": res["disease_confidence"],
            "correct_detector": detector_match,
            "correct_disease": disease_match,
            "is_error": is_error,
            "error_type": error_type
        })
        
        # Store confidences for distribution plots
        if pred_is_watermelon:
            confidences.append(res["disease_confidence"] if res["disease_confidence"] is not None else 0.0)
            
        if (idx+1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(image_files)} images...")
            
    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nBatch predictions saved to: {args.output_csv}")
    
    # Save Misclassified report
    with open(args.output_misclassified, "w") as f:
        f.write("=== BATCH EVALUATION ERROR ANALYSIS ===\n")
        f.write(f"Total Images: {len(image_files)}\n")
        f.write(f"Misclassified Count: {len(misclassified)}\n\n")
        for m in misclassified:
            f.write(f"File: {m['path']}\n")
            f.write(f"  Error Type: {m['error_type']}\n")
            f.write(f"  Status:     {m['status']}\n")
            f.write(f"  Detector Confidence: {m['watermelon_conf']:.2%}\n")
            dis_conf_str = f"{m['disease_conf']:.2%}" if m['disease_conf'] is not None else 'N/A'
            f.write(f"  Disease Confidence:  {dis_conf_str}\n")
            f.write("-" * 50 + "\n")
    print(f"Misclassification details saved to: {args.output_misclassified}")
    
    # Print high-level metrics
    detector_acc = correct_detector / len(image_files) if len(image_files) > 0 else 0
    disease_acc = correct_disease / disease_eligible if disease_eligible > 0 else 0
    print("\n" + "="*40)
    print("        BATCH EVALUATION RESULTS")
    print("="*40)
    print(f"Total Images Evaluated: {len(image_files)}")
    print(f"Detector Accuracy:      {detector_acc:.2%} ({correct_detector}/{len(image_files)})")
    print(f"Disease Class Accuracy: {disease_acc:.2%} ({correct_disease}/{disease_eligible})")
    print(f"Total Errors Found:     {len(misclassified)}")
    print("="*40 + "\n")
    
    # Generate Confidence Distribution Histogram
    if confidences:
        plt.figure(figsize=(8, 5))
        plt.hist(confidences, bins=10, range=(0, 1), edgecolor='black', color='skyblue', alpha=0.7)
        plt.axvline(x=analyzer.disease_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({analyzer.disease_threshold})')
        plt.title("Disease Classifier Confidence Distribution", fontsize=14, fontweight='bold')
        plt.xlabel("Confidence Score", fontsize=12)
        plt.ylabel("Number of Images", fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("reports/figures/confidence_distribution.png", dpi=150)
        plt.close()
        print("Confidence distribution plot saved to reports/figures/confidence_distribution.png")

if __name__ == "__main__":
    run_batch_evaluation()
