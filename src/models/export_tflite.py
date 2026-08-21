import os
import tensorflow as tf
import numpy as np
from src.data.data_loader import preprocess_image

def representative_dataset_gen_disease():
    """
    Generator of representative images from the processed disease train set
    to calibrate INT8 quantization activations.
    """
    train_dir = "data/processed/train"
    if not os.path.exists(train_dir):
        # Fallback if splits aren't prepared
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
        return
        
    count = 0
    # Collect 100 random images for calibration
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith('.jpg'):
                path = os.path.join(root, f)
                img, _ = preprocess_image(path, 0, augment=False)
                # Reshape to (1, 224, 224, 3)
                img_expanded = tf.expand_dims(img, axis=0)
                yield [img_expanded]
                count += 1
                if count >= 100:
                    return

def representative_dataset_gen_detector():
    """
    Generator of representative images from the detector train set.
    """
    train_dir = "data/detector/train"
    if not os.path.exists(train_dir):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
        return
        
    count = 0
    for root, _, files in os.walk(train_dir):
        for f in files:
            if f.lower().endswith('.jpg'):
                path = os.path.join(root, f)
                img, _ = preprocess_image(path, 0, augment=False)
                img_expanded = tf.expand_dims(img, axis=0)
                yield [img_expanded]
                count += 1
                if count >= 100:
                    return

def convert_to_tflite(keras_model_path, output_dir, name_prefix, representative_gen=None):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Converting Keras model {keras_model_path} to TFLite ---")
    model = tf.keras.models.load_model(keras_model_path)
    
    # 1. Float32 TFLite
    print("Converting to Float32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float32 = converter.convert()
    f32_path = os.path.join(output_dir, f"{name_prefix}_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_float32)
        
    # 2. Float16 Quantization
    print("Converting to Float16 Quantized TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_float16 = converter.convert()
    f16_path = os.path.join(output_dir, f"{name_prefix}_float16.tflite")
    with open(f16_path, "wb") as f:
        f.write(tflite_float16)
        
    # 3. Full INT8 Quantization (with representative dataset calibration)
    int8_path = os.path.join(output_dir, f"{name_prefix}_int8.tflite")
    if representative_gen is not None:
        print("Converting to INT8 Quantized TFLite (with calibration)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set input/output types as float32 for mobile compatibility (TFLite handles scaling internally)
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        try:
            tflite_int8 = converter.convert()
            with open(int8_path, "wb") as f:
                f.write(tflite_int8)
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        except Exception as e:
            print(f"INT8 conversion failed: {e}. Generating dynamic range INT8 fallback...")
            # Fallback to simple dynamic range quantization (no calibration needed)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_int8 = converter.convert()
            with open(int8_path, "wb") as f:
                f.write(tflite_int8)
            int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    else:
        # Dynamic range fallback
        print("Converting to Dynamic Range INT8 TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_int8 = converter.convert()
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    f32_size = os.path.getsize(f32_path) / (1024 * 1024)
    f16_size = os.path.getsize(f16_path) / (1024 * 1024)
    
    print(f"\nSize Comparison for '{name_prefix}':")
    print(f"  Float32 size: {f32_size:.2f} MB")
    print(f"  Float16 size: {f16_size:.2f} MB")
    print(f"  INT8 size:    {int8_size:.2f} MB")
    
    return {
        "float32_size_mb": f32_size,
        "float16_size_mb": f16_size,
        "int8_size_mb": int8_size
    }

def main():
    detector_keras = "models/detector/best_detector_model.h5"
    if not os.path.exists(detector_keras):
        detector_keras = "models/detector/final_detector_model.h5"
        
    disease_keras = "models/disease/best_disease_model.h5"
    if not os.path.exists(disease_keras):
        disease_keras = "models/disease/final_disease_model.h5"
        
    # Verify models exist
    if not os.path.exists(detector_keras) or not os.path.exists(disease_keras):
        print("Error: Models have not been trained yet. Please run training scripts first.")
        return
        
    detector_tflite_stats = convert_to_tflite(
        detector_keras, 
        "exports/detector", 
        "watermelon_detector", 
        representative_dataset_gen_detector
    )
    
    disease_tflite_stats = convert_to_tflite(
        disease_keras, 
        "exports/disease", 
        "watermelon_disease", 
        representative_dataset_gen_disease
    )
    
    # Save a comparison summary report
    summary = [
        "# TFLite Model Quantization & Size Comparison Report",
        "",
        "| Model Type | Quantization | Size (MB) | Compression Ratio |",
        "| :--- | :--- | :---: | :---: |"
    ]
    
    # Add detector stats
    d_f32 = detector_tflite_stats["float32_size_mb"]
    d_f16 = detector_tflite_stats["float16_size_mb"]
    d_int8 = detector_tflite_stats["int8_size_mb"]
    summary.append(f"| Watermelon Detector | Float32 | {d_f32:.2f} MB | 1.0x |")
    summary.append(f"| Watermelon Detector | Float16 | {d_f16:.2f} MB | {d_f32/d_f16:.2f}x |")
    summary.append(f"| Watermelon Detector | INT8 | {d_int8:.2f} MB | {d_f32/d_int8:.2f}x |")
    
    # Add disease stats
    c_f32 = disease_tflite_stats["float32_size_mb"]
    c_f16 = disease_tflite_stats["float16_size_mb"]
    c_int8 = disease_tflite_stats["int8_size_mb"]
    summary.append(f"| Disease Classifier | Float32 | {c_f32:.2f} MB | 1.0x |")
    summary.append(f"| Disease Classifier | Float16 | {c_f16:.2f} MB | {c_f32/c_f16:.2f}x |")
    summary.append(f"| Disease Classifier | INT8 | {c_int8:.2f} MB | {c_f32/c_int8:.2f}x |")
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/tflite_quantization_report.md", "w") as f:
        f.write("\n".join(summary))
    print("\nSaved TFLite Quantization Report to reports/tflite_quantization_report.md")

if __name__ == "__main__":
    main()
