import numpy as np
from pathlib import Path
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# Transaction Risk Score
# ============================================================

def run_xgboost(transaction_features, save_path=None):

    # --------------------------------------------------------
    # Build X and y
    # --------------------------------------------------------

    drop_columns = [
        "transaction_id",
        "nameOrig",
        "nameDest",
        "timestamp",
        "isFraud",
        "isMoneyLaundering",
        "laundering_typology",
    ]

    X = transaction_features.drop(
        columns=drop_columns,
        errors="ignore",
    )

    X = X.select_dtypes(
        include=np.number
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    y = transaction_features[
        "isFraud"
    ].astype(np.int8)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    order = transaction_features[
        "timestamp"
    ].sort_values().index

    n = len(order)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    positive = y_train.sum()
    negative = len(y_train) - positive

    scale_pos_weight = (
        negative / max(positive, 1)
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\n[TX-RISK] Training XGBoost..."
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val),
        ],
        verbose=False,
    )

    # --------------------------------------------------------
    # Test / validation risk scores
    # --------------------------------------------------------

    val_scores = model.predict_proba(
        X_val
    )[:, 1]

    test_scores = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    val_pr_auc = average_precision_score(
        y_val,
        val_scores,
    )

    val_roc_auc = roc_auc_score(
        y_val,
        val_scores,
    )

    # --------------------------------------------------------
    # Find threshold using validation F1
    # --------------------------------------------------------

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in thresholds:

        pred = (
            val_scores >= threshold
        ).astype(np.int8)

        f1 = f1_score(
            y_val,
            pred,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_pred = (
        test_scores >= best_threshold
    ).astype(np.int8)

    test_pr_auc = average_precision_score(
        y_test,
        test_scores,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        test_scores,
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n[TX-RISK] Score distribution:"
    )

    print(
        np.percentile(
            test_scores,
            [
                0,
                25,
                50,
                75,
                90,
                95,
                99,
                99.5,
                99.9,
                100,
            ],
        )
    )

    print(
        f"\n[TX-RISK] Actual fraud rate: "
        f"{y_test.mean():.4%}"
    )

    print(
        f"[TX-RISK] Predicted fraud rate: "
        f"{test_pred.mean():.4%}"
    )

    print(
        f"[TX-RISK] Predicted fraud count: "
        f"{test_pred.sum():,} / {len(test_pred):,}"
    )

    print(
        "\n[TX-RISK] Validation:"
    )

    print(
        f"PR-AUC:  {val_pr_auc:.4f}"
    )

    print(
        f"ROC-AUC: {val_roc_auc:.4f}"
    )

    print(
        f"Best threshold: {best_threshold:.2f}"
    )

    print(
        "\n[TX-RISK] Test:"
    )

    print(
        f"PR-AUC:   {test_pr_auc:.4f}"
    )

    print(
        f"ROC-AUC:  {test_roc_auc:.4f}"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall:    {test_recall:.4f}"
    )

    print(
        f"F1:        {test_f1:.4f}"
    )

    # ========================================================
    # Generate risk scores for ALL transactions
    #
    # IMPORTANT:
    # This uses the SAME fitted XGBoost model.
    #
    # It does NOT retrain the model.
    # It does NOT change test metrics.
    #
    # These scores are for downstream topology/ring analysis.
    # ========================================================

    print(
        "\n[TX-RISK] Scoring all transactions..."
    )

    all_scores = model.predict_proba(
        X
    )[:, 1]

    transaction_risk_all_results = (
        transaction_features[
            [
                "transaction_id",
                "nameOrig",
                "nameDest",
                "timestamp",
                "isFraud",
            ]
        ]
        .copy()
    )

    transaction_risk_all_results[
        "risk_score"
    ] = all_scores

    transaction_risk_all_results[
        "predicted_fraud"
    ] = (
        all_scores >= best_threshold
    ).astype(np.int8)

    print(
        f"[TX-RISK] Scored "
        f"{len(transaction_risk_all_results):,} "
        f"transactions"
    )

    # --------------------------------------------------------
    # Attach risk score to TEST transactions
    #
    # This remains your evaluation result.
    # --------------------------------------------------------

    results = transaction_features.loc[
        test_idx,
        [
            "transaction_id",
            "nameOrig",
            "nameDest",
            "timestamp",
            "isFraud",
        ],
    ].copy()

    results["risk_score"] = test_scores

    results["predicted_fraud"] = (
        test_scores >= best_threshold
    ).astype(np.int8)

    # --------------------------------------------------------
    # Save TEST transaction risk scores
    # --------------------------------------------------------

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_parquet(
            save_path,
            index=False,
        )

        print(
            f"[SAVE] Test transaction risk scores saved to "
            f"{save_path}"
        )

        # ----------------------------------------------------
        # Save ALL transaction risk scores
        # ----------------------------------------------------

        all_path = (
            save_path.parent
            / "transaction_risk_all.parquet"
        )

        transaction_risk_all_results.to_parquet(
            all_path,
            index=False,
        )

        print(
            f"[SAVE] All transaction risk scores saved to "
            f"{all_path}"
        )

    return (
        model,
        results,
        transaction_risk_all_results,
    )