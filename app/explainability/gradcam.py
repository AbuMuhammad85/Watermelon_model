import os
import streamlit as st
import numpy as np
import tensorflow as tf
from src.evaluation.grad_cam import get_gradcam_heatmap, overlay_heatmap

def run_gradcam_explanation(model, img_path, model_format="keras", last_conv_layer_name="out_relu"):
    """
    Wrapper for Grad-CAM explanation.
    Returns: overlay_image (numpy array), original_image (numpy array), heatmap (numpy array), or None if unsupported.
    """
    if model_format != "keras":
        # Grad-CAM requires a Keras model with symbolic graph backprop
        return None
        
    try:
        # Load and preprocess image
        img_bytes = tf.io.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_bytes, channels=3)
        img_resized = tf.image.resize(img_decoded, (224, 224))
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
        img_array = tf.expand_dims(img_preprocessed, axis=0)
        
        # Get heatmap
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
        
        # Overlay heatmap
        overlay, original = overlay_heatmap(img_path, heatmap)
        
        return {
            "overlay": overlay,
            "original": original,
            "heatmap": heatmap
        }
    except Exception as e:
        print(f"Grad-CAM execution failed: {e}")
        return None
