# src/models/node_risk.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.20
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4


# ============================================================
# MLP Model
# ============================================================

class NodeRiskMLP(nn.Module):
    """
    MLP for node/account-level risk prediction.

    Input:
        raw node features + GraphSAGE embedding

    Output:
        one fraud/risk logit per node
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.ReLU(),

            nn.LayerNorm(
                hidden_dim // 2
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(
            x
        ).squeeze(-1)


# ============================================================
# Build node-level MLP input
# ============================================================

def build_node_risk_dataset(
    node_features: pd.DataFrame,
    node_embeddings: torch.Tensor,
):
    """
    Build the combined node-level input.

    Combines:

        raw node features
        +
        GraphSAGE node embeddings

    Returns
    -------
    X:
        torch.Tensor [num_nodes, num_features]

    y:
        torch.Tensor [num_nodes]

    feature_names:
        List of feature names.
    """

    # ========================================================
    # 1. Validate node feature table
    # ========================================================

    required_columns = [
        "account_id",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in node_features.columns
    ]

    if missing:
        raise ValueError(
            f"node_features is missing columns: {missing}"
        )

    # ========================================================
    # 2. Validate embeddings
    # ========================================================

    if not isinstance(
        node_embeddings,
        torch.Tensor,
    ):
        node_embeddings = torch.tensor(
            np.asarray(
                node_embeddings
            ),
            dtype=torch.float32,
        )

    if node_embeddings.ndim != 2:
        raise ValueError(
            "node_embeddings must be a 2D tensor."
        )

    if len(node_features) != (
        node_embeddings.shape[0]
    ):
        raise ValueError(
            "Number of node feature rows does not match "
            "number of GraphSAGE embeddings."
        )

    # ========================================================
    # 3. Select raw node features
    # ========================================================

    excluded_columns = {
        "account_id",
        "label",
    }

    feature_columns = [
        column
        for column in node_features.columns
        if column not in excluded_columns
    ]

    raw_features = (
        node_features[
            feature_columns
        ]
        .select_dtypes(
            include=np.number
        )
        .copy()
    )

    if raw_features.empty:
        raise ValueError(
            "No numeric node features available "
            "for the MLP."
        )

    # ========================================================
    # 4. Clean raw node features
    # ========================================================

    raw_features = (
        raw_features
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    raw_tensor = torch.tensor(
        raw_features.to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    # ========================================================
    # 5. Clean GraphSAGE embeddings
    # ========================================================

    embeddings = (
        node_embeddings
        .detach()
        .cpu()
        .float()
    )

    embeddings = torch.nan_to_num(
        embeddings,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ========================================================
    # 6. Combine raw node features + embeddings
    # ========================================================

    X = torch.cat(
        [
            raw_tensor,
            embeddings,
        ],
        dim=1,
    )

    # ========================================================
    # 7. Build target
    # ========================================================

    y = torch.tensor(
        node_features["label"].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
    )

    # ========================================================
    # 8. Feature names
    # ========================================================

    embedding_dim = embeddings.shape[1]

    embedding_columns = [
        f"gnn_emb_{i}"
        for i in range(
            embedding_dim
        )
    ]

    feature_names = (
        list(raw_features.columns)
        + embedding_columns
    )

    # ========================================================
    # 9. Final validation
    # ========================================================

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y have different numbers of nodes."
        )

    if X.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "Number of feature names does not match X."
        )

    print(
        f"[NODE-RISK] Nodes: "
        f"{X.shape[0]:,}"
    )

    print(
        f"[NODE-RISK] Raw node features: "
        f"{raw_tensor.shape[1]}"
    )

    print(
        f"[NODE-RISK] GraphSAGE embedding dimension: "
        f"{embedding_dim}"
    )

    print(
        f"[NODE-RISK] Combined input dimension: "
        f"{X.shape[1]}"
    )

    print(
        f"[NODE-RISK] Positive nodes: "
        f"{int(y.sum().item()):,}"
    )

    return (
        X,
        y,
        feature_names,
    )


# ============================================================
# Normalize input
# ============================================================

def normalize_node_features(
    X: torch.Tensor,
    train_mask: torch.Tensor,
):
    """
    Standardize node features.

    IMPORTANT:
        The scaler is fitted ONLY on training nodes.

    Then the same scaler is applied to validation
    and test nodes.

    This prevents preprocessing leakage.
    """

    X_numpy = (
        X.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    train_mask_numpy = (
        train_mask
        .detach()
        .cpu()
        .numpy()
    )

    if train_mask_numpy.sum() == 0:
        raise ValueError(
            "Training mask contains no nodes."
        )

    scaler = StandardScaler()

    scaler.fit(
        X_numpy[
            train_mask_numpy
        ]
    )

    X_scaled = scaler.transform(
        X_numpy
    )

    X_scaled = np.asarray(
        X_scaled,
        dtype=np.float32,
    )

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_scaled = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    )

    print(
        "[NODE-RISK] Feature normalization complete."
    )

    return (
        X_scaled,
        scaler,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
):
    """
    Calculate node-level classification metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(np.int8)

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int8)

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    if len(np.unique(y_true)) > 1:

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

    else:

        pr_auc = float("nan")
        roc_auc = float("nan")

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Find best validation threshold
# ============================================================

def find_best_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
):
    """
    Find the threshold that maximizes validation F1.

    IMPORTANT:
        This threshold is selected on validation data only.
        It must not be tuned using the test set.
    """

    best_threshold = 0.5
    best_f1 = -1.0

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1,
    )


# ============================================================
# Train MLP
# ============================================================

def train_node_risk_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: str | None = None,
):
    """
    Train the node-risk MLP.

    Existing PyG train/validation masks are reused.
    No new split is created here.

    Model checkpoint selection:
        validation PR-AUC
    """

    # ========================================================
    # 1. Validate
    # ========================================================

    if len(X) != len(y):
        raise ValueError(
            "X and y must have the same number of nodes."
        )

    if len(train_mask) != len(X):
        raise ValueError(
            "train_mask does not match number of nodes."
        )

    if len(val_mask) != len(X):
        raise ValueError(
            "val_mask does not match number of nodes."
        )

    # ========================================================
    # 2. Device
    # ========================================================

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        device
    )

    print(
        f"[NODE-RISK] Device: {device}"
    )

    # ========================================================
    # 3. Normalize input
    # ========================================================

    print(
        "[NODE-RISK] Normalizing input features..."
    )

    X_scaled, scaler = (
        normalize_node_features(
            X,
            train_mask,
        )
    )

    # ========================================================
    # 4. Move tensors to device
    # ========================================================

    X_scaled = X_scaled.to(
        device=device,
        dtype=torch.float32,
    )

    y = y.to(
        device=device,
        dtype=torch.float32,
    )

    train_mask = train_mask.to(
        device
    )

    val_mask = val_mask.to(
        device
    )

    # ========================================================
    # 5. Build model
    # ========================================================

    model = NodeRiskMLP(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    # ========================================================
    # 6. Class imbalance
    # ========================================================

    train_positive = (
        y[train_mask]
        .sum()
        .item()
    )

    train_total = (
        train_mask
        .sum()
        .item()
    )

    train_negative = (
        train_total
        - train_positive
    )

    if train_positive <= 0:
        raise ValueError(
            "Training split contains no positive nodes."
        )

    if train_negative <= 0:
        raise ValueError(
            "Training split contains no negative nodes."
        )

    pos_weight_value = (
        train_negative
        / train_positive
    )

    pos_weight = torch.tensor(
        pos_weight_value,
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    print(
        f"[NODE-RISK] Train positives: "
        f"{int(train_positive):,}"
    )

    print(
        f"[NODE-RISK] Train negatives: "
        f"{int(train_negative):,}"
    )

    print(
        f"[NODE-RISK] Positive class weight: "
        f"{pos_weight_value:.4f}"
    )

    # ========================================================
    # 7. Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ========================================================
    # 8. Best model tracking
    # ========================================================

    best_val_pr_auc = -1.0
    best_val_loss = float("inf")
    best_state = None

    print(
        f"\n[NODE-RISK] Training "
        f"{epochs} epochs..."
    )

    # ========================================================
    # 9. Training loop
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()

        logits = model(
            X_scaled
        )

        train_loss = criterion(
            logits[train_mask],
            y[train_mask],
        )

        train_loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_scaled
            )

            val_loss = criterion(
                val_logits[val_mask],
                y[val_mask],
            )

            val_probabilities = (
                torch.sigmoid(
                    val_logits[val_mask]
                )
                .detach()
                .cpu()
                .numpy()
            )

            val_labels = (
                y[val_mask]
                .detach()
                .cpu()
                .numpy()
            )

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        if len(
            np.unique(
                val_labels
            )
        ) > 1:

            val_pr_auc = (
                average_precision_score(
                    val_labels,
                    val_probabilities,
                )
            )

            val_roc_auc = (
                roc_auc_score(
                    val_labels,
                    val_probabilities,
                )
            )

        else:

            val_pr_auc = float("nan")
            val_roc_auc = float("nan")

        # ----------------------------------------------------
        # Save best checkpoint based on PR-AUC
        # ----------------------------------------------------

        if (
            not np.isnan(val_pr_auc)
            and val_pr_auc > best_val_pr_auc
        ):

            best_val_pr_auc = (
                val_pr_auc
            )

            best_val_loss = (
                val_loss.item()
            )

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict()
                .items()
            }

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"Epoch "
            f"{epoch:03d}/{epochs} "
            f"| Train Loss: "
            f"{train_loss.item():.6f} "
            f"| Val Loss: "
            f"{val_loss.item():.6f} "
            f"| Val PR-AUC: "
            f"{val_pr_auc:.6f} "
            f"| Val ROC-AUC: "
            f"{val_roc_auc:.6f}"
        )

    # ========================================================
    # 10. Restore best model
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    print(
        f"\n[NODE-RISK] Best validation PR-AUC: "
        f"{best_val_pr_auc:.6f}"
    )

    print(
        f"[NODE-RISK] Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    return (
        model,
        scaler,
    )


# ============================================================
# Generate node risk scores
# ============================================================

def predict_node_risk_scores(
    model: NodeRiskMLP,
    X: torch.Tensor,
    scaler: StandardScaler,
    device: str | None = None,
):
    """
    Generate node-level risk scores for every node.

    The supplied scaler MUST be the scaler fitted on
    training nodes.
    """

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        device
    )

    # --------------------------------------------------------
    # Apply training-fitted scaler
    # --------------------------------------------------------

    X_numpy = (
        X.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    X_scaled = scaler.transform(
        X_numpy
    )

    X_scaled = np.nan_to_num(
        X_scaled,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_scaled = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    ).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    model = model.to(
        device
    )

    model.eval()

    with torch.no_grad():

        logits = model(
            X_scaled
        )

        risk_scores = torch.sigmoid(
            logits
        )

    return risk_scores.cpu()


# ============================================================
# Build node risk results
# ============================================================

def build_node_risk_results(
    node_features: pd.DataFrame,
    risk_scores: torch.Tensor,
):
    """
    Attach node risk scores to account IDs.
    """

    if len(node_features) != len(
        risk_scores
    ):
        raise ValueError(
            "Number of risk scores does not match "
            "number of nodes."
        )

    results = node_features[
        [
            "account_id",
            "label",
        ]
    ].copy()

    results["risk_score"] = (
        risk_scores
        .detach()
        .cpu()
        .numpy()
    )

    return results


# ============================================================
# Complete node-risk pipeline
# ============================================================

def run_node_risk_model(
    node_features: pd.DataFrame,
    node_embeddings: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: str | None = None,
):
    """
    Complete node-level risk pipeline.

    1. Build raw node features + GraphSAGE embeddings.
    2. Normalize using training nodes only.
    3. Train MLP using existing train/val masks.
    4. Restore best model based on validation PR-AUC.
    5. Generate risk scores for all nodes.
    6. Evaluate validation and test performance.
    """

    # ========================================================
    # 1. Build input
    # ========================================================

    print(
        "\n[NODE-RISK] Building MLP input..."
    )

    (
        X,
        y,
        feature_names,
    ) = build_node_risk_dataset(
        node_features=node_features,
        node_embeddings=node_embeddings,
    )

    # ========================================================
    # 2. Train
    # ========================================================

    (
        model,
        scaler,
    ) = train_node_risk_mlp(
        X=X,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # ========================================================
    # 3. Generate risk score for ALL nodes
    # ========================================================

    print(
        "\n[NODE-RISK] Calculating node risk scores..."
    )

    risk_scores = predict_node_risk_scores(
        model=model,
        X=X,
        scaler=scaler,
        device=device,
    )

    # ========================================================
    # 4. Build results
    # ========================================================

    results = build_node_risk_results(
        node_features=node_features,
        risk_scores=risk_scores,
    )

    # ========================================================
    # 5. Add split information
    # ========================================================

    train_mask_numpy = (
        train_mask
        .detach()
        .cpu()
        .numpy()
    )

    val_mask_numpy = (
        val_mask
        .detach()
        .cpu()
        .numpy()
    )

    test_mask_numpy = (
        test_mask
        .detach()
        .cpu()
        .numpy()
    )

    results["split"] = "unknown"

    results.loc[
        train_mask_numpy,
        "split",
    ] = "train"

    results.loc[
        val_mask_numpy,
        "split",
    ] = "validation"

    results.loc[
        test_mask_numpy,
        "split",
    ] = "test"

    # ========================================================
    # 6. Evaluate validation set
    # ========================================================

    all_probabilities = (
        risk_scores
        .detach()
        .cpu()
        .numpy()
    )

    y_numpy = (
        y
        .detach()
        .cpu()
        .numpy()
    )

    val_probabilities = (
        all_probabilities[
            val_mask_numpy
        ]
    )

    val_labels = (
        y_numpy[
            val_mask_numpy
        ]
    )

    val_metrics = calculate_metrics(
        y_true=val_labels,
        probabilities=val_probabilities,
        threshold=0.5,
    )

    # --------------------------------------------------------
    # Best threshold from validation
    # --------------------------------------------------------

    (
        best_threshold,
        best_val_f1,
    ) = find_best_threshold(
        val_labels,
        val_probabilities,
    )

    val_metrics_best_threshold = (
        calculate_metrics(
            y_true=val_labels,
            probabilities=val_probabilities,
            threshold=best_threshold,
        )
    )

    # ========================================================
    # 7. Evaluate test set
    # ========================================================

    test_probabilities = (
        all_probabilities[
            test_mask_numpy
        ]
    )

    test_labels = (
        y_numpy[
            test_mask_numpy
        ]
    )

    test_metrics = calculate_metrics(
        y_true=test_labels,
        probabilities=test_probabilities,
        threshold=best_threshold,
    )

    # ========================================================
    # 8. Print validation metrics
    # ========================================================

    print(
        "\n[NODE-RISK] Validation metrics "
        "at threshold 0.50:"
    )

    print(
        f"  PR-AUC:  "
        f"{val_metrics['pr_auc']:.6f}"
    )

    print(
        f"  ROC-AUC: "
        f"{val_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Precision:"
        f" {val_metrics['precision']:.6f}"
    )

    print(
        f"  Recall:   "
        f"{val_metrics['recall']:.6f}"
    )

    print(
        f"  F1:       "
        f"{val_metrics['f1']:.6f}"
    )

    print(
        "\n[NODE-RISK] Best validation threshold:"
        f" {best_threshold:.2f}"
    )

    print(
        f"[NODE-RISK] Validation F1 at best "
        f"threshold: {best_val_f1:.6f}"
    )

    # ========================================================
    # 9. Print test metrics
    # ========================================================

    print(
        "\n[NODE-RISK] Test metrics "
        f"at validation-selected threshold "
        f"{best_threshold:.2f}:"
    )

    print(
        f"  PR-AUC:  "
        f"{test_metrics['pr_auc']:.6f}"
    )

    print(
        f"  ROC-AUC: "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Precision:"
        f" {test_metrics['precision']:.6f}"
    )

    print(
        f"  Recall:   "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"  F1:       "
        f"{test_metrics['f1']:.6f}"
    )

    # ========================================================
    # 10. Summary
    # ========================================================

    print(
        "\n[NODE-RISK] Completed."
    )

    print(
        f"[NODE-RISK] Input shape: "
        f"{tuple(X.shape)}"
    )

    print(
        f"[NODE-RISK] Risk scores generated: "
        f"{len(risk_scores):,}"
    )

    print(
        f"[NODE-RISK] Score range: "
        f"{risk_scores.min().item():.6f} → "
        f"{risk_scores.max().item():.6f}"
    )

    # ========================================================
    # 11. Highest-risk accounts
    # ========================================================

    print(
        "\n[NODE-RISK] Highest-risk nodes:"
    )

    print(
        results
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        [
            [
                "account_id",
                "risk_score",
                "label",
                "split",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Return
    # ========================================================

    metrics = {
        "validation": val_metrics,
        "validation_best_threshold": (
            val_metrics_best_threshold
        ),
        "test": test_metrics,
        "best_threshold": best_threshold,
    }

    return (
        model,
        results,
        X,
        y,
        feature_names,
        scaler,
        metrics,
    )


# ============================================================
# Optional standalone execution
# ============================================================

if __name__ == "__main__":

    print(
        "node_risk.py defines the node-risk MLP pipeline."
    )

    print(
        "Run it through src.pipeline.py so it receives "
        "the existing node features, GraphSAGE embeddings, "
        "and PyG train/validation/test masks."
    )