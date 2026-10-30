"""
Transaction-level tabular fraud risk scoring using XGBoost.

Trains a gradient-boosted decision tree ensemble on tabular transaction features,
calibrated with class-imbalance weighting, chronological splitting, and threshold tuning.
Produces transaction risk scores for test evaluation and downstream topology analysis.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

# Ensure src/ is on sys.path for direct CLI execution
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import evaluate_predictions, print_evaluation_report

DROP_COLUMNS: list[str] = [
    "transaction_id",
    "nameOrig",
    "nameDest",
    "timestamp",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
]


def prepare_feature_matrix(
    transaction_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract numeric feature matrix X and target y from transaction features."""
    X = (
        transaction_features.drop(columns=DROP_COLUMNS, errors="ignore")
        .select_dtypes(include=np.number)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = transaction_features["isFraud"].astype(np.int8)
    return X, y


def run_xgboost(
    transaction_features: pd.DataFrame,
    save_path: str | Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Train and evaluate XGBoost transaction fraud detector.

    Parameters
    ----------
    transaction_features : pd.DataFrame
        Complete transaction features including timestamps and labels.
    save_path : str or Path, optional
        Path to save test evaluation scores and complete transaction risk scores.
    train_ratio : float, default 0.70
    val_ratio : float, default 0.15

    Returns
    -------
    model : xgb.XGBClassifier
        Fitted XGBoost model.
    test_results : pd.DataFrame
        Test set transactions with predicted probabilities and labels.
    all_results : pd.DataFrame
        Complete transaction dataset with predicted risk scores for downstream topology mining.
    """
    X, y = prepare_feature_matrix(transaction_features)

    # 1. Chronological Train / Val / Test Split
    order = transaction_features["timestamp"].sort_values().index
    n = len(order)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # 2. Scale positive weight for extreme fraud class imbalance
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"\n[TX-RISK] Training XGBoost (Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,})...")
    print(f"  • Positive fraud cases in train: {pos_count:,} (scale_pos_weight: {scale_pos_weight:.2f})")

    # 3. Model Training
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # 4. Predict risk scores
    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    # 5. Threshold Optimization & Validation Metrics
    val_metrics = evaluate_predictions(y_val.to_numpy(), val_scores)
    best_threshold = val_metrics.threshold
    print_evaluation_report("TX-RISK VAL", val_metrics, y_val.to_numpy(), val_scores)

    # 6. Test Set Evaluation
    test_metrics = evaluate_predictions(y_test.to_numpy(), test_scores, threshold=best_threshold)
    print_evaluation_report("TX-RISK TEST", test_metrics, y_test.to_numpy(), test_scores)

    # 7. Build Full Transaction Risk Table for Downstream Topology Mining
    print("\n[TX-RISK] Scoring all transactions for downstream graph analysis...")
    all_scores = model.predict_proba(X)[:, 1]

    base_cols = ["transaction_id", "nameOrig", "nameDest", "timestamp", "isFraud"]
    all_results = transaction_features[base_cols].copy()
    all_results["risk_score"] = all_scores
    all_results["predicted_fraud"] = (all_scores >= best_threshold).astype(np.int8)

    # 8. Test Set Evaluation Artifact
    test_results = transaction_features.loc[test_idx, base_cols].copy()
    test_results["risk_score"] = test_scores
    test_results["predicted_fraud"] = (test_scores >= best_threshold).astype(np.int8)

    # 9. Save Artifacts if Path Provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        test_results.to_parquet(save_path, index=False)
        print(f"[SAVE] Test transaction risk scores saved to: {save_path}")

        all_path = save_path.parent / "transaction_risk_all.parquet"
        all_results.to_parquet(all_path, index=False)
        print(f"[SAVE] Complete transaction risk scores saved to: {all_path}")

    return model, test_results, all_results


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    DATA_PATH = Path("artifacts/transaction_features.parquet")
    if DATA_PATH.exists():
        print("Testing Transaction_risk_score.py on cached transaction features...")
        df_tx = pd.read_parquet(DATA_PATH).head(10000)
        m, test_res, all_res = run_xgboost(df_tx)
        print(f"Test results shape: {test_res.shape}")
        print(f"All results shape:  {all_res.shape}")
        print("XGBoost risk module verification passed successfully!")
    else:
        print(f"Artifact {DATA_PATH} not found. Run pipeline to generate.")
