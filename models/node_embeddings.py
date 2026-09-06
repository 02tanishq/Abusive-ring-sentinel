# src/models/graphsage.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.optim as optim


class GraphSAGEEncoder(nn.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.norm1 = nn.LayerNorm(
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            embedding_dim,
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

        # Used only for node-level training.
        # The downstream risk model will be in another file.
        self.classifier = nn.Linear(
            embedding_dim,
            1,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate node embeddings.

        Returns:
            Tensor of shape:
                [num_nodes, embedding_dim]
        """

        x = self.conv1(
            x,
            edge_index,
        )

        x = self.norm1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index,
        )

        x = self.norm2(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:

            embeddings:
                Node embeddings.

            logits:
                Node classification logits.
        """

        embeddings = self.encode(
            x,
            edge_index,
        )

        logits = self.classifier(
            embeddings
        ).squeeze(-1)

        return embeddings, logits


def train_graphsage(
    data,
    hidden_channels: int = 128,
    embedding_dim: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
):
    """
    Train GraphSAGE for node-level suspicious-account
    classification.

    Expected data:
        data.x
        data.edge_index
        data.y
        data.train_mask
        data.val_mask

    Returns:
        model
        best_embeddings
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    data = data.to(device)

    model = GraphSAGEEncoder(
        in_channels=data.x.size(1),
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Class imbalance handling
    # --------------------------------------------------------

    train_labels = data.y[
        data.train_mask
    ]

    positive_count = (
        train_labels == 1
    ).sum().float()

    negative_count = (
        train_labels == 0
    ).sum().float()

    if positive_count == 0:
        raise ValueError(
            "Training split contains no positive nodes."
        )

    pos_weight = (
        negative_count
        / positive_count
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        optimizer.zero_grad()

        embeddings, logits = model(
            data.x,
            data.edge_index,
        )

        train_logits = logits[
            data.train_mask
        ]

        train_labels = data.y[
            data.train_mask
        ].float()

        loss = F.binary_cross_entropy_with_logits(
            train_logits,
            train_labels,
            pos_weight=pos_weight,
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            _, val_logits = model(
                data.x,
                data.edge_index,
            )

            val_logits = val_logits[
                data.val_mask
            ]

            val_labels = data.y[
                data.val_mask
            ].float()

            val_loss = (
                F.binary_cross_entropy_with_logits(
                    val_logits,
                    val_labels,
                    pos_weight=pos_weight,
                )
            )

        if val_loss.item() < best_val_loss:

            best_val_loss = (
                val_loss.item()
            )

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.state_dict().items()
                )
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a valid model."
        )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Generate final node embeddings
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():
        embeddings, _ = model(
            data.x,
            data.edge_index,
        )

    return model, embeddings.cpu()

print("it_is_good")