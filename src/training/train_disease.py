import os
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.data_loader import get_dataset
from src.models.disease_classifier import build_disease_model

def get_class_weights(train_dir):
    """
    Computes class weights dynamically based on class counts in train_dir.
    Matches the index sorting of os.listdir.
    """
    class_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    counts = [len(os.listdir(os.path.join(train_dir, d))) for d in class_dirs]
    total = sum(counts)
    num_classes = len(class_dirs)
    
    class_weights = {}
    print("Dataset counts per disease class:")
    for idx, (name, count) in enumerate(zip(class_dirs, counts)):
        weight = total / (num_classes * count) if count > 0 else 1.0
        class_weights[idx] = weight
        print(f"  Class {idx} ('{name}'): count={count}, weight={weight:.4f}")
        
    return class_weights

def plot_history(history, output_dir, prefix="disease"):
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
    processed_dir = "data/processed"
    model_save_dir = "models/disease"
    figures_dir = "reports/figures"
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Batch size 16 to avoid OOM on 8GB RAM CPU
    batch_size = 16
    
    print("Loading datasets...")
    train_dataset, class_names = get_dataset(
        os.path.join(processed_dir, 'train'),
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    val_dataset, _ = get_dataset(
        os.path.join(processed_dir, 'val'),
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )
    
    # Save the labels mapping for Flutter integration later
    labels_file = os.path.join(model_save_dir, "disease_labels.txt")
    with open(labels_file, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved disease labels to {labels_file}")
    
    # Build model
    print("Building model...")
    model, base_model = build_disease_model(num_classes=len(class_names))
    
    # Callbacks
    checkpoint_filepath = os.path.join(model_save_dir, "best_disease_model.h5")
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
    
    # Compute class weights to mitigate imbalance
    class_weights = get_class_weights(os.path.join(processed_dir, 'train'))
    
    # ------------------ STAGE A: Frozen Backbone ------------------
    print("\n--- STAGE A: Training Classification Head ---")
    history_a = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # ------------------ STAGE B: Fine Tuning ------------------
    print("\n--- STAGE B: Fine-Tuning Backbone ---")
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze BatchNormalization layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    # Recompile with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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
    model.save(os.path.join(model_save_dir, "final_disease_model.h5"))
    print("Disease model saved successfully.")
    
    # Plot curves
    plot_history(history_a, figures_dir, prefix="disease_stage_a")
    plot_history(history_b, figures_dir, prefix="disease_stage_b")

if __name__ == "__main__":
    main()
