import os
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_keras_model(model_path):
    """
    Load a Keras H5 model with caching.
    """
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading Keras model from {model_path}: {e}")
        return None

@st.cache_resource
def load_tflite_interpreter(model_path):
    """
    Load a TFLite model and allocate tensors with caching.
    """
    if not os.path.exists(model_path):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model from {model_path}: {e}")
        return None
