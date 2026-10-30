"""
Unified evaluation and threshold optimization metrics for Abuse Ring Sentinel.

Provides shared performance evaluation routines for tabular (XGBoost), graph
neural network (GraphSAGE + MLP), and ring topology alert scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Evaluation summary for binary fraud risk classification."""

    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01,
) -> Tuple[float, float]:
    """
    Find decision threshold that maximizes F1 score on validation predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_prob : np.ndarray
        Predicted continuous risk probabilities in [0, 1].
    step : float, default 0.01
        Grid search step size.

    Returns
    -------
    best_threshold : float
        Optimal probability cutoff.
    best_f1 : float
        Highest F1 achieved at optimal threshold.
    """
    thresholds = np.arange(step, 1.0, step)
    best_threshold: float = 0.50
    best_f1: float = 0.0

    for th in thresholds:
        preds = (y_prob >= th).astype(np.int8)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(th)

    return best_threshold, best_f1


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> ClassificationMetrics:
    """
    Compute comprehensive binary classification metrics.

    If threshold is None, searches for the threshold that maximizes F1.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels.
    y_prob : np.ndarray
        Predicted continuous risk scores.
    threshold : float, optional
        Predefined decision threshold.

    Returns
    -------
    ClassificationMetrics
        Container with PR-AUC, ROC-AUC, threshold, precision, recall, and F1.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Calculate rank-based ranking metrics (threshold-invariant)
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    if threshold is None:
        threshold, _ = find_best_threshold(y_true, y_prob)

    preds = (y_prob >= threshold).astype(np.int8)
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    return ClassificationMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def print_evaluation_report(
    prefix: str,
    metrics: ClassificationMetrics,
    y_true: np.ndarray | None = None,
    y_prob: np.ndarray | None = None,
) -> None:
    """Print a clean, standardized evaluation summary for console and logs."""
    print(f"\n[{prefix}] Performance Report (Threshold: {metrics.threshold:.2f}):")
    print(f"  • PR-AUC:    {metrics.pr_auc:.4f}")
    print(f"  • ROC-AUC:   {metrics.roc_auc:.4f}")
    print(f"  • Precision: {metrics.precision:.4f}")
    print(f"  • Recall:    {metrics.recall:.4f}")
    print(f"  • F1 Score:  {metrics.f1:.4f}")

    if y_prob is not None:
        percentiles = [50, 90, 95, 99, 99.9]
        vals = np.percentile(y_prob, percentiles)
        pct_str = " | ".join(f"p{p}: {v:.3f}" for p, v in zip(percentiles, vals))
        print(f"  • Risk Percentiles: {pct_str}")


if __name__ == "__main__":
    # Self-test
    y_mock = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    prob_mock = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.4, 0.7, 0.2, 0.95, 0.3])
    res = evaluate_predictions(y_mock, prob_mock)
    print_evaluation_report("TEST", res, y_mock, prob_mock)
    print("Metrics module verified!")
