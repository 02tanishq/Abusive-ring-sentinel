"""
Account/node-level fraud risk prediction module for Abuse Ring Sentinel.

Combines raw tabular account features with learned GraphSAGE structural node
embeddings into a deep Multilayer Perceptron (MLP). Trains with leakage-free feature
standardization and class imbalance weighting to produce account-level risk scores.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure src/ is on sys.path for direct CLI execution
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import evaluate_predictions, print_evaluation_report


class NodeRiskMLP(nn.Module):
    """
    Multilayer Perceptron for account-level fraud classification.

    Architecture:
        Linear(input_dim, hidden_dim) -> ReLU -> LayerNorm -> Dropout
        Linear(hidden_dim, hidden_dim // 2) -> ReLU -> LayerNorm -> Dropout
        Linear(hidden_dim // 2, 1) -> Sigmoid / Logit
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing 1D classification logits."""
        return self.network(x).squeeze(-1)


def build_node_risk_dataset(
    node_features: pd.DataFrame,
    node_embeddings: torch.Tensor | np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Concatenate raw account tabular features with GraphSAGE embeddings.

    Parameters
    ----------
    node_features : pd.DataFrame
        Account features containing 'account_id' and 'label'.
    node_embeddings : torch.Tensor or np.ndarray
        GraphSAGE representations of shape [num_nodes, embedding_dim].

    Returns
    -------
    X : torch.Tensor [num_nodes, total_features]
    y : torch.Tensor [num_nodes]
    feature_names : list[str]
    """
    if "account_id" not in node_features.columns or "label" not in node_features.columns:
        raise ValueError("node_features must contain 'account_id' and 'label'.")

    # 1. Prepare raw tabular features
    raw_feature_cols = [
        c for c in node_features.columns
        if c not in {"account_id", "label"}
    ]
    raw_numeric = (
        node_features[raw_feature_cols]
        .select_dtypes(include=np.number)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    raw_tensor = torch.tensor(raw_numeric.to_numpy(dtype=np.float32), dtype=torch.float32)

    # 2. Prepare embedding tensor
    if isinstance(node_embeddings, np.ndarray):
        emb_tensor = torch.from_numpy(node_embeddings.astype(np.float32))
    else:
        emb_tensor = node_embeddings.detach().cpu().float()

    emb_tensor = torch.nan_to_num(emb_tensor, nan=0.0)

    if raw_tensor.shape[0] != emb_tensor.shape[0]:
        raise ValueError(
            f"Shape mismatch: {raw_tensor.shape[0]} node feature rows vs "
            f"{emb_tensor.shape[0]} embeddings."
        )

    # 3. Concatenate tabular and GNN features
    X = torch.cat([raw_tensor, emb_tensor], dim=1)
    y = torch.tensor(node_features["label"].to_numpy(dtype=np.float32), dtype=torch.float32)

    emb_dim = emb_tensor.shape[1]
    emb_cols = [f"gnn_emb_{i}" for i in range(emb_dim)]
    feature_names = list(raw_numeric.columns) + emb_cols

    return X, y, feature_names


def normalize_node_features(
    X: torch.Tensor,
    train_mask: torch.Tensor,
) -> Tuple[torch.Tensor, StandardScaler]:
    """
    Standardize feature representations.

    Critically fits StandardScaler ONLY on train_mask nodes to prevent
    information leakage from validation and test sets.
    """
    X_np = X.detach().cpu().numpy().astype(np.float32)
    mask_np = train_mask.detach().cpu().numpy().astype(bool)

    if mask_np.sum() == 0:
        raise ValueError("train_mask contains no nodes to fit scaler.")

    scaler = StandardScaler()
    scaler.fit(X_np[mask_np])
    X_scaled = np.nan_to_num(scaler.transform(X_np).astype(np.float32), nan=0.0)

    return torch.from_numpy(X_scaled), scaler


def train_node_risk_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    hidden_dim: int = 128,
    dropout: float = 0.20,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device | None = None,
) -> Tuple[NodeRiskMLP, StandardScaler]:
    """Train NodeRiskMLP on normalized features with class-weighted binary cross entropy."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Leakage-free standardization
    X_scaled, scaler = normalize_node_features(X, train_mask)

    X_scaled = X_scaled.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    model = NodeRiskMLP(
        input_dim=X_scaled.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Class imbalance weighting
    train_labels = y[train_mask]
    pos_count = (train_labels == 1).sum()
    neg_count = (train_labels == 0).sum()
    pos_weight = neg_count / torch.clamp(pos_count, min=1.0)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    print(f"\n[NODE-RISK] Training MLP on {device} ({epochs} epochs, pos_weight: {pos_weight:.2f})...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(X_scaled)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask],
            train_labels,
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_logits = model(X_scaled)[val_mask]
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits,
                y[val_mask],
                pos_weight=pos_weight,
            )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

    if best_state is None:
        raise RuntimeError("Node risk training failed to produce a valid model.")

    model.load_state_dict(best_state)
    return model, scaler


def run_node_risk_model(
    node_features: pd.DataFrame,
    node_embeddings: torch.Tensor | np.ndarray,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    hidden_dim: int = 128,
    dropout: float = 0.20,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device | None = None,
) -> Tuple[NodeRiskMLP, pd.DataFrame, torch.Tensor, torch.Tensor, List[str], StandardScaler, Any]:
    """
    End-to-end orchestration for account-level node risk scoring.

    Trains MLP, tunes optimal threshold on validation set, evaluates test metrics,
    and returns account risk predictions dataframe.
    """
    X, y, feature_names = build_node_risk_dataset(node_features, node_embeddings)

    model, scaler = train_node_risk_mlp(
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

    # Predict risk probabilities across all accounts
    dev = next(model.parameters()).device
    X_scaled_all = torch.from_numpy(
        np.nan_to_num(scaler.transform(X.numpy()).astype(np.float32), nan=0.0)
    ).to(dev)

    model.eval()
    with torch.no_grad():
        logits_all = model(X_scaled_all)
        scores_all = torch.sigmoid(logits_all).cpu().numpy()

    y_np = y.numpy()
    val_mask_np = val_mask.cpu().numpy().astype(bool)
    test_mask_np = test_mask.cpu().numpy().astype(bool)

    # Validation evaluation & threshold tuning
    val_metrics = evaluate_predictions(y_np[val_mask_np], scores_all[val_mask_np])
    print_evaluation_report("NODE-RISK VAL", val_metrics, y_np[val_mask_np], scores_all[val_mask_np])

    # Test set evaluation
    test_metrics = evaluate_predictions(y_np[test_mask_np], scores_all[test_mask_np], threshold=val_metrics.threshold)
    print_evaluation_report("NODE-RISK TEST", test_metrics, y_np[test_mask_np], scores_all[test_mask_np])

    # Build results table matching canonical contract
    split_labels = np.full(len(node_features), "train", dtype=object)
    split_labels[val_mask_np] = "val"
    split_labels[test_mask_np] = "test"

    results_df = pd.DataFrame({
        "account_id": node_features["account_id"],
        "label": node_features["label"].astype(int),
        "risk_score": scores_all,
        "split": split_labels,
    })

    return model, results_df, X, y, feature_names, scaler, test_metrics


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    NODE_FEATS = Path("artifacts/node_features.parquet")
    EMB_FEATS = Path("artifacts/node_embeddings.pt")

    if NODE_FEATS.exists() and EMB_FEATS.exists():
        print("Testing node_risk_score.py on cached node features & embeddings...")
        nodes_df = pd.read_parquet(NODE_FEATS)
        embeddings = torch.load(EMB_FEATS, weights_only=False)

        n = len(nodes_df)
        t_mask = torch.zeros(n, dtype=torch.bool)
        v_mask = torch.zeros(n, dtype=torch.bool)
        te_mask = torch.zeros(n, dtype=torch.bool)
        t_mask[:int(n * 0.7)] = True
        v_mask[int(n * 0.7):int(n * 0.85)] = True
        te_mask[int(n * 0.85):] = True

        m, res, X_t, y_t, feats, sc, met = run_node_risk_model(
            node_features=nodes_df,
            node_embeddings=embeddings,
            train_mask=t_mask,
            val_mask=v_mask,
            test_mask=te_mask,
            epochs=10,
        )
        print(f"Node risk results: {res.shape}")
        print(res.head(3))
        print("Node risk module verified successfully!")
    else:
        print("Required artifacts not found for standalone testing.")
