# src/risk_features.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


# ============================================================
# Columns that must NOT be used as XGBoost features
# ============================================================

NON_FEATURE_COLUMNS = {
    "transaction_id",
    "nameOrig",
    "nameDest",
    "timestamp",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
}


# ============================================================
# Build combined transaction + GraphSAGE feature dataset
# ============================================================

def build_transaction_risk_dataset(
    transaction_features: pd.DataFrame,
    node_embeddings: torch.Tensor,
    node_ids,
):
    """
    Combine transaction-level features with sender and
    receiver GraphSAGE embeddings.

    Returns
    -------
    risk_features : pd.DataFrame
        Complete transaction-level table including IDs,
        labels, transaction features, and embeddings.

    X : pd.DataFrame
        Numeric feature matrix for XGBoost.

    y : pd.Series
        Fraud target.
    """

    # --------------------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "transaction_id",
        "nameOrig",
        "nameDest",
        "isFraud",
        "timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in transaction_features.columns
    ]

    if missing:
        raise ValueError(
            f"transaction_features is missing: {missing}"
        )

    # --------------------------------------------------------
    # 2. Validate embedding shape
    # --------------------------------------------------------

    if isinstance(
        node_embeddings,
        torch.Tensor,
    ):
        if node_embeddings.ndim != 2:
            raise ValueError(
                "node_embeddings must be a 2D tensor."
            )

        embeddings = (
            node_embeddings
            .detach()
            .cpu()
            .numpy()
        )

    else:

        embeddings = np.asarray(
            node_embeddings
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "node_embeddings must be a 2D array."
            )

    if len(node_ids) != embeddings.shape[0]:
        raise ValueError(
            "Number of node_ids does not match "
            "number of node embeddings."
        )

    embedding_dim = embeddings.shape[1]

    # --------------------------------------------------------
    # 3. Build account -> embedding index
    # --------------------------------------------------------

    node_to_idx = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }

    # --------------------------------------------------------
    # 4. Map sender and receiver accounts
    # --------------------------------------------------------

    sender_idx = (
        transaction_features["nameOrig"]
        .map(node_to_idx)
    )

    receiver_idx = (
        transaction_features["nameDest"]
        .map(node_to_idx)
    )

    # --------------------------------------------------------
    # 5. Check mapping
    # --------------------------------------------------------

    if sender_idx.isna().any():

        missing_senders = (
            transaction_features.loc[
                sender_idx.isna(),
                "nameOrig",
            ]
            .unique()
        )

        raise ValueError(
            "Sender accounts missing from GraphSAGE "
            f"embeddings. Examples: "
            f"{missing_senders[:10]}"
        )

    if receiver_idx.isna().any():

        missing_receivers = (
            transaction_features.loc[
                receiver_idx.isna(),
                "nameDest",
            ]
            .unique()
        )

        raise ValueError(
            "Receiver accounts missing from GraphSAGE "
            f"embeddings. Examples: "
            f"{missing_receivers[:10]}"
        )

    sender_idx = (
        sender_idx
        .astype(np.int64)
        .to_numpy()
    )

    receiver_idx = (
        receiver_idx
        .astype(np.int64)
        .to_numpy()
    )

    # --------------------------------------------------------
    # 6. Retrieve embeddings
    # --------------------------------------------------------

    sender_embeddings = embeddings[
        sender_idx
    ]

    receiver_embeddings = embeddings[
        receiver_idx
    ]

    # --------------------------------------------------------
    # 7. Create embedding columns
    # --------------------------------------------------------

    sender_columns = [
        f"sender_emb_{i}"
        for i in range(embedding_dim)
    ]

    receiver_columns = [
        f"receiver_emb_{i}"
        for i in range(embedding_dim)
    ]

    sender_df = pd.DataFrame(
        sender_embeddings,
        columns=sender_columns,
    )

    receiver_df = pd.DataFrame(
        receiver_embeddings,
        columns=receiver_columns,
    )

    # --------------------------------------------------------
    # 8. Combine transaction features + embeddings
    # --------------------------------------------------------

    risk_features = pd.concat(
        [
            transaction_features.reset_index(
                drop=True
            ),
            sender_df,
            receiver_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # 9. Build target
    # --------------------------------------------------------

    y = (
        risk_features["isFraud"]
        .astype(np.int8)
        .copy()
    )

    # --------------------------------------------------------
    # 10. Build X
    # --------------------------------------------------------

    X = risk_features.drop(
        columns=[
            column
            for column in NON_FEATURE_COLUMNS
            if column in risk_features.columns
        ]
    )

    # Only numerical columns go into XGBoost
    X = X.select_dtypes(
        include=np.number
    )

    # Clean numerical values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # 11. Final validation
    # --------------------------------------------------------

    if len(X) != len(y):
        raise ValueError(
            "X and y have different numbers of rows."
        )

    if X.isna().any().any():
        raise ValueError(
            "X still contains NaN values."
        )

    print(
        f"[RISK] Transactions: "
        f"{len(X):,}"
    )

    print(
        f"[RISK] XGBoost features: "
        f"{X.shape[1]}"
    )

    print(
        f"[RISK] Embedding dimension: "
        f"{embedding_dim}"
    )

    print(
        f"[RISK] Sender embedding columns: "
        f"{len(sender_columns)}"
    )

    print(
        f"[RISK] Receiver embedding columns: "
        f"{len(receiver_columns)}"
    )

    print(
        f"[RISK] Fraud transactions: "
        f"{int(y.sum()):,}"
    )

    return (
        risk_features,
        X,
        y,
    )


# ============================================================
# Chronological transaction split
# ============================================================

def chronological_split(
    risk_features: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
):
    """
    Chronologically split transactions into train,
    validation, and test sets.
    """

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not 0 <= val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "train_ratio + val_ratio must be < 1."
        )

    # --------------------------------------------------------
    # Sort transactions by timestamp
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        risk_features["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "risk_features contains invalid timestamps."
        )

    order = (
        timestamps
        .sort_values()
        .index
    )

    n = len(order)

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (
            train_ratio + val_ratio
        )
    )

    train_idx = order[:train_end]
    val_idx = order[
        train_end:val_end
    ]
    test_idx = order[
        val_end:
    ]

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]

    X_val = X.loc[val_idx]
    y_val = y.loc[val_idx]

    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    print(
        f"[RISK] Train transactions: "
        f"{len(X_train):,}"
    )

    print(
        f"[RISK] Validation transactions: "
        f"{len(X_val):,}"
    )

    print(
        f"[RISK] Test transactions: "
        f"{len(X_test):,}"
    )

    print(
        f"[RISK] Train fraud: "
        f"{int(y_train.sum()):,}"
    )

    print(
        f"[RISK] Validation fraud: "
        f"{int(y_val.sum()):,}"
    )

    print(
        f"[RISK] Test fraud: "
        f"{int(y_test.sum()):,}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_idx,
        val_idx,
        test_idx,
    )