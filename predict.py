import os
import argparse
import json
from src.inference.pipeline import WatermelonAnalyzer
from src.evaluation.grad_cam import generate_and_save_gradcam

def main():
    parser = argparse.ArgumentParser(description="Watermelon AI - Image Prediction Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM activation map")
    parser.add_argument("--output_gradcam", type=str, default="reports/figures/gradcam_output.png", help="Path to save Grad-CAM visualization")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' does not exist.")
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
        
    # Initialize analyzer
    analyzer = WatermelonAnalyzer()
    
    # Run analysis
    result = analyzer.analyze(args.image)
    
    print("\n" + "="*40)
    print("        WATERMELON AI ANALYSIS")
    print("="*40)
    
    # Print human-readable output
    if result["status"] == "not_watermelon":
        print(f"Watermelon: NO")
        print(f"Confidence: {result['watermelon_confidence']:.2%}")
        print(f"Status:     REJECTED")
        print(f"Reason:     Image is outside the supported watermelon leaf domain.")
    elif result["status"] == "uncertain":
        print(f"Watermelon: YES")
        print(f"Watermelon Confidence: {result['watermelon_confidence']:.2%}")
        print(f"Disease:    Unknown (Ambiguous)")
        print(f"Disease Confidence:    {result['disease_confidence']:.2%}")
        print(f"Status:     UNCERTAIN")
        print(f"Reason:     Model is not confident in the specific disease class.")
    elif result["status"] == "confident":
        print(f"Watermelon: YES")
        print(f"Watermelon Confidence: {result['watermelon_confidence']:.2%}")
        print(f"Disease:    {result['disease']}")
        print(f"Disease Confidence:    {result['disease_confidence']:.2%}")
        print(f"Status:     CONFIDENT")
        print(f"Reason:     Successful disease classification.")
        
    print("-"*40)
    print("Structured JSON Result:")
    print(json.dumps(result, indent=2))
    print("="*40 + "\n")
    
    # Generate Grad-CAM if requested and valid watermelon
    if args.gradcam:
        if result["is_watermelon"]:
            print(f"Generating Grad-CAM visualization...")
            os.makedirs(os.path.dirname(args.output_gradcam), exist_ok=True)
            try:
                generate_and_save_gradcam(
                    analyzer.disease_classifier,
                    args.image,
                    args.output_gradcam
                )
                print(f"Grad-CAM saved to: {args.output_gradcam}")
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
        else:
            print("Grad-CAM skipped because image is not recognized as a watermelon leaf.")

if __name__ == "__main__":
    main()
