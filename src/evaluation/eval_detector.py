import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.data.data_loader import get_dataset, load_dataset_paths_labels, preprocess_image

def plot_confusion_matrix(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
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
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

def evaluate_detector():
    # Paths
    detector_dataset_dir = "data/detector"
    model_path = "models/detector/best_detector_model.h5"
    figures_dir = "reports/figures"
    
    if not os.path.exists(model_path):
        model_path = "models/detector/final_detector_model.h5"
        
    print(f"Loading detector model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Loading test dataset...")
    # Load test dataset
    file_paths, labels, class_names = load_dataset_paths_labels(os.path.join(detector_dataset_dir, 'test'))
    
    # Process images manually to keep same order
    images = []
    for path in file_paths:
        img, _ = preprocess_image(path, 0, augment=False)
        images.append(img)
    
    X_test = tf.stack(images)
    y_true = np.array(labels)
    
    print("Running predictions...")
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate Metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr_curve, tpr_curve)
    
    print("\n=== BINARY DETECTOR TEST PERFORMANCE ===")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f} (Sensitivity)")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    
    # Plot Confusion Matrix
    plot_confusion_matrix(cm, class_names, title="Detector Confusion Matrix", normalize=False)
    plt.savefig(os.path.join(figures_dir, "detector_cm.png"), dpi=150)
    plt.close()
    
    # Plot ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "detector_roc.png"), dpi=150)
    plt.close()
    
    # ------------------ OOD Rejection Evaluation ------------------
    ood_dir = "data/ood_test"
    if os.path.exists(ood_dir):
        print("\n=== OUT-OF-DOMAIN (OOD) REJECTION EVALUATION ===")
        ood_categories = [d for d in os.listdir(ood_dir) if os.path.isdir(os.path.join(ood_dir, d))]
        
        report_lines = [
            "# Out-of-Domain Rejection Report",
            "",
            "| OOD Category | Total Images | Accepted as Watermelon | Rejected | False Acceptance Rate |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ]
        
        total_ood_images = 0
        total_false_accepts = 0
        
        for cat in ood_categories:
            cat_dir = os.path.join(ood_dir, cat)
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith('.jpg')]
            if not files:
                continue
                
            cat_images = []
            for f in files:
                img_path = os.path.join(cat_dir, f)
                img, _ = preprocess_image(img_path, 0, augment=False)
                cat_images.append(img)
                
            X_cat = tf.stack(cat_images)
            preds_probs = model.predict(X_cat).flatten()
            
            # Predict as positive (watermelon) if prob >= 0.5
            accepted = np.sum(preds_probs >= 0.5)
            rejected = len(preds_probs) - accepted
            far = accepted / len(preds_probs)
            
            report_lines.append(f"| {cat} | {len(preds_probs)} | {accepted} | {rejected} | {far:.2%} |")
            print(f"Category: {cat:<20} | Total: {len(preds_probs):<3} | Accepted: {accepted:<3} | Rejected: {rejected:<3} | FAR: {far:.2%}")
            
            total_ood_images += len(preds_probs)
            total_false_accepts += accepted
            
        overall_far = total_false_accepts / total_ood_images if total_ood_images > 0 else 0
        report_lines.append(f"| **Overall** | **{total_ood_images}** | **{total_false_accepts}** | **{total_ood_images - total_false_accepts}** | **{overall_far:.2%}** |")
        
        # Save Report
        with open("reports/ood_rejection_report.md", "w") as f:
            f.write("\n".join(report_lines))
        print(f"\nSaved OOD Rejection Report to reports/ood_rejection_report.md (Overall FAR: {overall_far:.2%})")

if __name__ == "__main__":
    evaluate_detector()
