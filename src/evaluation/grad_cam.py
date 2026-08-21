import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_gradcam_heatmap(model, img_array, last_conv_layer_name="out_relu", pred_index=None):
    """
    Computes the Grad-CAM heatmap for a given image, supporting nested Keras models.
    """
    # Force Keras to trace symbolic inputs/outputs by running a forward pass
    _ = model(img_array)
    
    # Find the nested base model (backbone) and its index
    base = model
    base_idx = 0
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            base = layer
            base_idx = idx
            break
            
    try:
        # Build a gradient model mapping from backbone input to the target conv layer activation and backbone output
        last_conv_tensor = base.get_layer(last_conv_layer_name).output
        grad_model = tf.keras.models.Model(
            inputs=[base.input],
            outputs=[last_conv_tensor, base.output]
        )
    except Exception as e:
        print(f"Error building backbone grad model with layer '{last_conv_layer_name}': {e}. Attempting automated search...")
        # Fallback search for a 4D conv layer inside base
        last_conv_layer = None
        for layer in reversed(base.layers):
            try:
                shape = getattr(layer, 'output_shape', None)
                if shape and len(shape) == 4:
                    last_conv_layer = layer
                    break
            except Exception:
                continue
                
        if last_conv_layer is None:
            raise ValueError("Could not find a suitable 4D convolutional layer for Grad-CAM.")
        
        print(f"Using auto-selected layer: {last_conv_layer.name}")
        grad_model = tf.keras.models.Model(
            inputs=[base.input],
            outputs=[last_conv_layer.output, base.output]
        )

    # Record gradients of the class score with respect to intermediate activations in the tape
    with tf.GradientTape() as tape:
        # 1. Forward pass through the backbone model
        last_conv_layer_output, backbone_output = grad_model(img_array)
        
        # 2. Forward pass through the remaining top layers of the outer model
        x = backbone_output
        for layer in model.layers[base_idx + 1:]:
            x = layer(x)
        preds = x
        
        # 3. Select target class prediction
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Select target channel score
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            # Binary classification (sigmoid output)
            class_channel = preds[:, 0]
        else:
            # Multi-class classification (softmax output)
            class_channel = preds[:, pred_index]

    # Compute gradients of the class score with respect to target feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the target feature map channels by their pooled gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive activations
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Superimposes heatmap onto original image using Matplotlib colormaps and PIL.
    """
    # Load original image using PIL
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Rescale heatmap and apply colormap
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(heatmap)
    jet_heatmap = jet_colors[:, :, :3]
    
    # Resize heatmap to match image size
    jet_heatmap_img = Image.fromarray(np.uint8(255 * jet_heatmap))
    jet_heatmap_img = jet_heatmap_img.resize((224, 224), resample=Image.BILINEAR)
    jet_heatmap_array = np.array(jet_heatmap_img)
    
    # Blend images
    superimposed_img = jet_heatmap_array * alpha + img_array * (1.0 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img, img_array

def generate_and_save_gradcam(model, img_path, save_path, last_conv_layer_name="out_relu"):
    """
    Generates Grad-CAM and saves side-by-side visualization.
    """
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
    
    # Plot side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM Activation Map", fontsize=12, fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
