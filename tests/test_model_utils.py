import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# from src.model_utils import evaluate_model

def test_evaluate_model_outputs():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_proba = [0.2, 0.8, 0.4, 0.3, 0.9]

    metrics = evaluate_model(y_true, y_pred, y_proba)

    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"])
    assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1
