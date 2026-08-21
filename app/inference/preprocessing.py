import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for MobileNetV2.
    Expects input 'image' to be a PIL Image or path or file-like object.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    else:
        image = image.convert('RGB')
        
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    
    # MobileNetV2 preprocessing expects pixels scaled between -1 and 1
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_tensor = np.expand_dims(img_preprocessed, axis=0)
    return img_tensor
