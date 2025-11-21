# app/ML_framework_classification/core/evaluator.py
# ENTERPRISE-GRADE EVALUATION MODULE – FULL METRICS + CONFUSION MATRIX + ROC-AUC

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Optional
import json

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Complete evaluation for classification models
    Supports binary and multiclass
    """
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2
    is_multiclass = n_classes > 2

    # Base metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    result = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }

    # ROC-AUC
    if y_prob is not None:
        try:
            if is_binary:
                auc_score = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                result["roc_auc"] = round(float(auc_score), 4)
            elif is_multiclass:
                # One-vs-rest AUC
                auc_ovr = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                result["roc_auc_ovr_macro"] = round(float(auc_ovr), 4)
        except Exception as e:
            result["roc_auc_error"] = str(e)

    # Confusion Matrix (as list of lists for JSON)
    cm = confusion_matrix(y_true, y_pred).tolist()
    result["confusion_matrix"] = cm

    # Full classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cleaned_report = {}
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            cleaned_report[str(label)] = {
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1-score"], 4),
                "support": int(metrics["support"])
            }
        else:
            cleaned_report[label] = round(float(metrics), 4)
    result["classification_report"] = cleaned_report

    return result