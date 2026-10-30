"""
GraphSAGE representation learning module for Abuse Ring Sentinel.

Trains an inductive GraphSAGE graph neural network to learn 64-dimensional structural
account embeddings from transaction connectivity and raw account features.
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """
    Inductive 2-layer GraphSAGE GNN with LayerNorm and Dropout.

    Learns account embeddings by sampling and aggregating localized transaction
    neighborhoods. Outputs 64-dim structural account embeddings and binary logits.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Classification head for pre-training structural representations
        self.classifier = nn.Linear(embedding_dim, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through graph convolutions to extract node embeddings."""
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both embeddings and classification logits.

        Returns
        -------
        embeddings : torch.Tensor of shape [num_nodes, embedding_dim]
        logits : torch.Tensor of shape [num_nodes]
        """
        embeddings = self.encode(x, edge_index)
        logits = self.classifier(embeddings).squeeze(-1)
        return embeddings, logits


def train_graphsage(
    data: Data,
    hidden_channels: int = 128,
    embedding_dim: int = 64,
    dropout: float = 0.20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    device: str | torch.device | None = None,
) -> Tuple[GraphSAGEEncoder, torch.Tensor]:
    """
    Train GraphSAGE on account fraud labels with class-weighted loss.

    Restores model weights from the epoch with the lowest validation loss
    and extracts the final node embeddings.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG dataset containing x, edge_index, y, train_mask, val_mask.
    hidden_channels : int, default 128
    embedding_dim : int, default 64
    dropout : float, default 0.20
    lr : float, default 1e-3
    weight_decay : float, default 1e-4
    epochs : int, default 100
    device : torch.device, optional

    Returns
    -------
    model : GraphSAGEEncoder
        Best fitted GNN encoder.
    best_embeddings : torch.Tensor
        Extracted 64-dim embeddings of shape [num_nodes, embedding_dim].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    data = data.to(device)
    model = GraphSAGEEncoder(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Calculate class imbalance weighting on training set
    train_labels = data.y[data.train_mask]
    pos_count = (train_labels == 1).sum().float()
    neg_count = (train_labels == 0).sum().float()
    pos_weight = neg_count / torch.clamp(pos_count, min=1.0)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    print(f"\n[GNN] Training GraphSAGE on {device} ({epochs} epochs, pos_weight: {pos_weight:.2f})...")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        _, logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[data.train_mask],
            data.y[data.train_mask].float(),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            _, val_logits = model(data.x, data.edge_index)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits[data.val_mask],
                data.y[data.val_mask].float(),
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
        raise RuntimeError("GNN training failed to find a valid checkpoint.")

    model.load_state_dict(best_state)

    # Extract final canonical embeddings
    model.eval()
    with torch.no_grad():
        final_embeddings, _ = model(data.x, data.edge_index)

    return model, final_embeddings.cpu()


if __name__ == "__main__":
    # Self-test on synthetic graph data
    print("Self-testing GraphSAGEEncoder module...")
    mock_x = torch.randn(100, 16)
    mock_edge_index = torch.randint(0, 100, (2, 300))
    mock_y = torch.randint(0, 2, (100,)).float()
    mock_mask = torch.zeros(100, dtype=torch.bool)
    mock_mask[:70] = True
    mock_val = torch.zeros(100, dtype=torch.bool)
    mock_val[70:85] = True

    mock_data = Data(x=mock_x, edge_index=mock_edge_index, y=mock_y)
    mock_data.train_mask = mock_mask
    mock_data.val_mask = mock_val

    gnn_model, emb = train_graphsage(mock_data, epochs=10)
    print(f"Extracted Embedding Shape: {emb.shape}")
    assert emb.shape == (100, 64)
    print("GraphSAGE self-test passed successfully!")
