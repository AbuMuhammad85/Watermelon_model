import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.data.data_loader import load_dataset_paths_labels, preprocess_image

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True):
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Greens')

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, ha='right')
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=12)

def evaluate_disease_model():
    # Paths
    processed_dir = "data/processed"
    model_path = "models/disease/best_disease_model.h5"
    figures_dir = "reports/figures"
    
    if not os.path.exists(model_path):
        model_path = "models/disease/final_disease_model.h5"
        
    print(f"Loading disease model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test dataset...")
    # Load test dataset
    file_paths, labels, class_names = load_dataset_paths_labels(os.path.join(processed_dir, 'test'))
    
    # Process images manually to keep same order
    images = []
    for path in file_paths:
        img, _ = preprocess_image(path, 0, augment=False)
        images.append(img)
        
    X_test = tf.stack(images)
    y_true = np.array(labels)
    
    print("Running predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n=== DISEASE CLASSIFIER TEST PERFORMANCE ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Save text report
    os.makedirs(os.path.dirname(figures_dir), exist_ok=True)
    with open("reports/disease_classification_report.txt", "w") as f:
        f.write("=== DISEASE CLASSIFIER TEST PERFORMANCE ===\n")
        f.write(f"Accuracy:    {accuracy:.4f}\n")
        f.write(f"Macro F1:    {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, title="Disease Classifier Confusion Matrix", normalize=False)
    plt.savefig(os.path.join(figures_dir, "disease_cm.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved under reports/figures/disease_cm.png")

if __name__ == "__main__":
    evaluate_disease_model()
