import os

def load_evaluation_metrics():
    """
    Parses pre-calculated evaluation files in reports/.
    Returns a dictionary of metrics, report strings, or None if missing.
    """
    metrics = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "raw_report": "",
        "ood_report": "",
        "quantization_report": ""
    }
    
    # 1. Parse disease classification report
    report_path = "reports/disease_classification_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            lines = f.readlines()
            metrics["raw_report"] = "".join(lines)
            for line in lines:
                if "Accuracy:" in line:
                    metrics["accuracy"] = float(line.split(":")[-1].strip())
                elif "Macro F1:" in line:
                    metrics["macro_f1"] = float(line.split(":")[-1].strip())
                elif "Weighted F1:" in line:
                    metrics["weighted_f1"] = float(line.split(":")[-1].strip())
                    
    # 2. Load OOD rejection report
    ood_path = "reports/ood_rejection_report.md"
    if os.path.exists(ood_path):
        with open(ood_path, "r") as f:
            metrics["ood_report"] = f.read()
            
    # 3. Load Quantization report
    quant_path = "reports/tflite_quantization_report.md"
    if os.path.exists(quant_path):
        with open(quant_path, "r") as f:
            metrics["quantization_report"] = f.read()
            
    return metrics
