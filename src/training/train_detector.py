import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.data_loader import get_dataset
from src.models.detector import build_detector_model

def get_class_weights(train_dir):
    """
    Computes class weights dynamically based on file counts.
    Class 0 = not_watermelon, Class 1 = watermelon.
    """
    neg_count = len(os.listdir(os.path.join(train_dir, 'not_watermelon')))
    pos_count = len(os.listdir(os.path.join(train_dir, 'watermelon')))
    total = neg_count + pos_count
    
    # Standard inverse frequency formula
    weight_0 = total / (2.0 * neg_count)
    weight_1 = total / (2.0 * pos_count)
    
    print(f"Dataset imbalance: Positives={pos_count}, Negatives={neg_count}")
    print(f"Computed weights: not_watermelon={weight_0:.4f}, watermelon={weight_1:.4f}")
    return {0: weight_0, 1: weight_1}

def plot_history(history, output_dir, prefix="detector"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff7f0e', linewidth=2)
    plt.title(f"{prefix.capitalize()} Accuracy", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_accuracy.png"), dpi=150)
    plt.close()

    # 2. Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='#d62728', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#2ca02c', linewidth=2)
    plt.title(f"{prefix.capitalize()} Loss", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_loss.png"), dpi=150)
    plt.close()

def main():
    # Paths
    detector_dataset_dir = "data/detector"
    model_save_dir = "models/detector"
    figures_dir = "reports/figures"
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Batch size 16 to avoid OOM on 8GB RAM CPU
    batch_size = 16
    
    print("Loading datasets...")
    train_dataset, class_names = get_dataset(
        os.path.join(detector_dataset_dir, 'train'),
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    val_dataset, _ = get_dataset(
        os.path.join(detector_dataset_dir, 'val'),
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    # Build model
    print("Building model...")
    model, base_model = build_detector_model()
    
    # Callbacks
    checkpoint_filepath = os.path.join(model_save_dir, "best_detector_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Compute class weights
    class_weights = get_class_weights(os.path.join(detector_dataset_dir, 'train'))
    
    # ------------------ STAGE A: Frozen Backbone ------------------
    print("\n--- STAGE A: Training Classification Head ---")
    history_a = model.fit(
        train_dataset,
        epochs=8,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # ------------------ STAGE B: Fine Tuning ------------------
    print("\n--- STAGE B: Fine-Tuning Backbone ---")
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze BatchNormalization layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    # Recompile with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    # Train stage B
    history_b = model.fit(
        train_dataset,
        epochs=5,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save the final fine-tuned model
    model.save(os.path.join(model_save_dir, "final_detector_model.h5"))
    print("Detector model saved successfully.")
    
    # Plot curves
    plot_history(history_a, figures_dir, prefix="detector_stage_a")
    plot_history(history_b, figures_dir, prefix="detector_stage_b")
    
if __name__ == "__main__":
    main()
