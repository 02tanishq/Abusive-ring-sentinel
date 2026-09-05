# src/pyg_builder.py

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx


def networkx_to_pyg(
    graph: nx.MultiDiGraph,
    node_features: pd.DataFrame,
    transaction_features: pd.DataFrame,
) -> Data:
    """
    Convert the master NetworkX graph into a PyTorch Geometric Data object.

    Graph structure:
        - NetworkX node = account/entity
        - NetworkX edge = individual transaction

    PyG outputs:
        data.x
            Node feature matrix.

        data.edge_index
            Source/destination node indices.

        data.edge_attr
            Transaction/edge feature matrix.

        data.y
            Node-level labels.

        data.node_ids
            Original account IDs.

        data.edge_ids
            Original transaction IDs.

    Notes:
        - IDs are NOT used as model features.
        - The same master graph created by graph_builder.py is used.
        - The graph is not rebuilt here.
    """

    # ============================================================
    # 1. Validate node features
    # ============================================================

    if "account_id" not in node_features.columns:
        raise ValueError(
            "node_features must contain 'account_id'."
        )

    # ============================================================
    # 2. Validate transaction features
    # ============================================================

    if "transaction_id" not in transaction_features.columns:
        raise ValueError(
            "transaction_features must contain "
            "'transaction_id'."
        )

    # ============================================================
    # 3. Get node ordering from the master graph
    # ============================================================

    node_to_idx = graph.graph.get(
        "node_to_idx"
    )

    if node_to_idx is None:
        raise ValueError(
            "Graph does not contain 'node_to_idx'. "
            "Use build_graph() from graph_builder.py."
        )

    # Reconstruct the exact graph ordering.
    node_ids = list(graph.nodes)

    num_nodes = len(node_ids)

    # Make sure mapping agrees with graph node order.
    for idx, node_id in enumerate(node_ids):
        if node_to_idx.get(node_id) != idx:
            raise ValueError(
                "Graph node ordering and node_to_idx mapping "
                "are inconsistent."
            )

    # ============================================================
    # 4. Node feature matrix
    # ============================================================

    node_lookup = (
        node_features
        .set_index("account_id")
        .to_dict(orient="index")
    )

    # Only numerical feature columns.
    node_feature_columns = [
        column
        for column in node_features.columns
        if column != "account_id"
    ]

    node_rows = []

    for node_id in node_ids:

        if node_id not in node_lookup:
            raise ValueError(
                f"No node features found for account: {node_id}"
            )

        row = node_lookup[node_id]

        values = []

        for column in node_feature_columns:

            value = row[column]

            if pd.isna(value):
                value = 0.0

            values.append(float(value))

        node_rows.append(values)

    x = torch.tensor(
        np.asarray(
            node_rows,
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )

    # ============================================================
    # 5. Node labels
    # ============================================================

    # The node feature table should contain a node-level target.
    #
    # If your current build_account_features() does not yet have
    # a label, create it separately before calling this function.

    if "label" in node_features.columns:

        node_labels = (
            node_features
            .set_index("account_id")
            .reindex(node_ids)["label"]
            .fillna(0)
            .astype(np.float32)
            .to_numpy()
        )

    elif "isFraud" in node_features.columns:

        node_labels = (
            node_features
            .set_index("account_id")
            .reindex(node_ids)["isFraud"]
            .fillna(0)
            .astype(np.float32)
            .to_numpy()
        )

    else:
        raise ValueError(
            "node_features must contain either "
            "'label' or 'isFraud' for node-level GNN training."
        )

    y = torch.tensor(
        node_labels,
        dtype=torch.float32,
    )

    # ============================================================
    # 6. Build edge_index
    # ============================================================

    edge_sources = []
    edge_destinations = []
    edge_ids = []

    for source, destination, key, edge_data in graph.edges(
        keys=True,
        data=True,
    ):

        if source not in node_to_idx:
            raise ValueError(
                f"Unknown source node: {source}"
            )

        if destination not in node_to_idx:
            raise ValueError(
                f"Unknown destination node: {destination}"
            )

        edge_sources.append(
            node_to_idx[source]
        )

        edge_destinations.append(
            node_to_idx[destination]
        )

        transaction_id = edge_data.get(
            "transaction_id",
            key,
        )

        edge_ids.append(
            transaction_id
        )

    edge_index = torch.tensor(
        [
            edge_sources,
            edge_destinations,
        ],
        dtype=torch.long,
    )

    # ============================================================
    # 7. Edge / transaction feature matrix
    # ============================================================

    transaction_lookup = (
        transaction_features
        .set_index("transaction_id")
        .to_dict(orient="index")
    )

    transaction_feature_columns = [
        column
        for column in transaction_features.columns
        if column not in {
            "transaction_id",
            "nameOrig",
            "nameDest",
            "timestamp",
            "isFraud",
            "isMoneyLaundering",
            "laundering_typology",
        }
    ]

    edge_rows = []

    for transaction_id in edge_ids:

        if transaction_id not in transaction_lookup:
            raise ValueError(
                "No transaction features found for "
                f"transaction_id={transaction_id}"
            )

        row = transaction_lookup[transaction_id]

        values = []

        for column in transaction_feature_columns:

            value = row[column]

            if pd.isna(value):
                value = 0.0

            # Edge features must be numerical.
            if not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(
                    f"Edge feature '{column}' contains "
                    f"non-numeric value: {value!r}"
                )

            values.append(float(value))

        edge_rows.append(values)

    edge_attr = torch.tensor(
        np.asarray(
            edge_rows,
            dtype=np.float32,
        ),
        dtype=torch.float32,
    )

    # ============================================================
    # 8. Create PyG Data object
    # ============================================================

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    # Keep IDs separately for tracing predictions back to
    # real accounts and transactions.
    data.node_ids = node_ids
    data.edge_ids = edge_ids

    return data

def create_temporal_node_masks(
    pyg_data,
    df,
    node_ids,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Create chronological train/validation/test masks for nodes.

    A node is ordered by the timestamp of its first
    observed transaction.
    """

    # --------------------------------------------------------
    # First transaction time for every account
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    temp = pd.DataFrame({
        "nameOrig": df["nameOrig"],
        "nameDest": df["nameDest"],
        "timestamp": timestamps,
    })

    outgoing = (
        temp[
            ["nameOrig", "timestamp"]
        ]
        .rename(
            columns={
                "nameOrig": "account_id"
            }
        )
    )

    incoming = (
        temp[
            ["nameDest", "timestamp"]
        ]
        .rename(
            columns={
                "nameDest": "account_id"
            }
        )
    )

    account_times = pd.concat(
        [outgoing, incoming],
        ignore_index=True,
    )

    first_seen = (
        account_times
        .groupby("account_id")["timestamp"]
        .min()
    )

    # --------------------------------------------------------
    # Align with PyG node ordering
    # --------------------------------------------------------

    node_times = pd.Series(
        node_ids
    ).map(first_seen)

    # Nodes without valid timestamps go last
    node_times = node_times.fillna(
        pd.Timestamp.max
    )

    order = (
        node_times
        .sort_values()
        .index
        .to_numpy()
    )

    num_nodes = len(order)

    train_end = int(
        num_nodes * train_ratio
    )

    val_end = int(
        num_nodes * (
            train_ratio + val_ratio
        )
    )

    train_idx = order[:train_end]
    val_idx = order[
        train_end:val_end
    ]
    test_idx = order[val_end:]

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    train_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    val_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    test_mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask

    return pyg_data