"""
PyTorch Geometric (PyG) conversion module for Abuse Ring Sentinel.

Converts the canonical NetworkX MultiDiGraph and pandas feature tables into a
PyG `Data` object containing:
    - data.x: Account node feature matrix [num_nodes, node_features]
    - data.edge_index: Graph connectivity tensor [2, num_edges]
    - data.edge_attr: Transaction feature tensor [num_edges, edge_features]
    - data.y: Account-level ground truth label [num_nodes]
    - data.train_mask, data.val_mask, data.test_mask: Chronological node splits
    - data.node_ids, data.edge_ids: Stable tracking identifiers
"""

from __future__ import annotations

from typing import List, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Columns in transaction features that are identifiers or targets (not edge features)
NON_EDGE_FEATURE_COLUMNS: set[str] = {
    "transaction_id",
    "nameOrig",
    "nameDest",
    "timestamp",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
}


def networkx_to_pyg(
    graph: nx.MultiDiGraph,
    node_features: pd.DataFrame,
    transaction_features: pd.DataFrame,
) -> Data:
    """
    Convert the master NetworkX graph into a PyTorch Geometric Data object.

    Uses high-speed vectorized NumPy tensor conversions to avoid slow Python dictionary
    lookups over millions of edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Master transaction graph containing canonical 'node_to_idx'.
    node_features : pd.DataFrame
        Account feature table with 'account_id' and label column ('label' or 'isFraud').
    transaction_features : pd.DataFrame
        Transaction feature table with 'transaction_id'.

    Returns
    -------
    torch_geometric.data.Data
        PyG Data object ready for GNN training.
    """
    # 1. Validate inputs
    if "account_id" not in node_features.columns:
        raise ValueError("node_features must contain an 'account_id' column.")
    if "transaction_id" not in transaction_features.columns:
        raise ValueError("transaction_features must contain a 'transaction_id' column.")

    node_to_idx = graph.graph.get("node_to_idx")
    if node_to_idx is None:
        raise ValueError("Graph missing 'node_to_idx'. Build via graph_builder.py first.")

    node_ids: list[str] = list(graph.nodes)
    num_nodes = len(node_ids)

    # 2. Vectorized Node Feature Matrix (x) and Target (y)
    node_df = node_features.set_index("account_id").reindex(node_ids)
    node_feature_cols = [
        col for col in node_features.columns
        if col not in {"account_id", "label", "isFraud"}
    ]

    x_np = (
        node_df[node_feature_cols]
        .select_dtypes(include=np.number)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    x = torch.from_numpy(np.ascontiguousarray(x_np).copy())

    label_col = "label" if "label" in node_df.columns else "isFraud"
    if label_col not in node_df.columns:
        raise ValueError("node_features must contain either 'label' or 'isFraud'.")

    y_np = (
        pd.to_numeric(node_df[label_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    y = torch.from_numpy(np.ascontiguousarray(y_np).copy())

    # 3. Vectorized Graph Topology (edge_index and edge_ids)
    edge_tuples = list(graph.edges(keys=True, data=True))
    sources = [node_to_idx[u] for u, v, k, d in edge_tuples]
    destinations = [node_to_idx[v] for u, v, k, d in edge_tuples]
    edge_ids = [int(d.get("transaction_id", k)) for u, v, k, d in edge_tuples]

    edge_index = torch.tensor([sources, destinations], dtype=torch.long)

    # 4. Vectorized Edge Feature Matrix (edge_attr)
    edge_feature_cols = [
        col for col in transaction_features.columns
        if col not in NON_EDGE_FEATURE_COLUMNS
    ]

    tx_numeric = (
        transaction_features.set_index("transaction_id")[edge_feature_cols]
        .select_dtypes(include=np.number)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    edge_attr_np = (
        tx_numeric.loc[edge_ids]
        .to_numpy(dtype=np.float32)
    )
    edge_attr = torch.from_numpy(np.ascontiguousarray(edge_attr_np).copy())

    # 5. Assemble PyG Data container
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    data.node_ids = node_ids
    data.edge_ids = edge_ids

    # Verification assertions
    assert data.num_nodes == num_nodes, f"Node count mismatch: {data.num_nodes} vs {num_nodes}"
    assert data.edge_index.shape[1] == len(edge_ids), "Edge count mismatch in edge_index."
    assert data.edge_attr.shape[0] == len(edge_ids), "Edge count mismatch in edge_attr."

    return data


def create_temporal_node_masks(
    pyg_data: Data,
    df: pd.DataFrame,
    node_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Data:
    """
    Create chronological train, validation, and test boolean masks for accounts.

    An account is ordered chronologically by the timestamp of its first observed
    transaction, preventing future data leakage into the GNN encoder.

    Parameters
    ----------
    pyg_data : Data
        PyG Data object to attach masks to.
    df : pd.DataFrame
        Transactions with 'nameOrig', 'nameDest', and 'timestamp'.
    node_ids : list[str]
        Canonical node ordering of the graph.
    train_ratio : float, default 0.70
        Fraction of earliest accounts for training.
    val_ratio : float, default 0.15
        Fraction of subsequent accounts for validation.

    Returns
    -------
    Data
        The updated PyG Data object with train_mask, val_mask, and test_mask attached.
    """
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios: train_ratio + val_ratio must be < 1.")

    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    tx_times = pd.DataFrame({
        "orig": df["nameOrig"],
        "dest": df["nameDest"],
        "time": timestamps,
    })

    # Find earliest observation for each account across origin and destination roles
    orig_first = tx_times.groupby("orig")["time"].min()
    dest_first = tx_times.groupby("dest")["time"].min()
    first_seen = pd.concat([orig_first, dest_first]).groupby(level=0).min()

    # Align with graph canonical node ordering
    node_times = pd.Series(node_ids).map(first_seen).fillna(pd.Timestamp.max)
    order = node_times.sort_values().index.to_numpy()

    num_nodes = len(order)
    train_end = int(num_nodes * train_ratio)
    val_end = int(num_nodes * (train_ratio + val_ratio))

    train_idx = order[:train_end].copy()
    val_idx = order[train_end:val_end].copy()
    test_idx = order[val_end:].copy()

    # Construct boolean tensor masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask

    return pyg_data


if __name__ == "__main__":
    from pathlib import Path
    from ingest import load_transactions
    from graph_builder import build_graph

    DATA_PATH = Path("data/raw/amlnet/AMLNet_v2_transactions.csv")
    if DATA_PATH.exists():
        print("Testing pyg_builder.py on 5,000 transaction sample...")
        df_sample = load_transactions(DATA_PATH, nrows=5000)
        graph, n2i, i2n = build_graph(df_sample)

        # Mock minimal features for testing
        node_feats = pd.DataFrame({
            "account_id": list(graph.nodes),
            "feat_1": np.random.randn(len(graph.nodes)).astype(np.float32),
            "label": np.random.choice([0, 1], size=len(graph.nodes), p=[0.95, 0.05]).astype(np.float32),
        })

        tx_feats = df_sample[["transaction_id", "nameOrig", "nameDest", "amount"]].copy()
        tx_feats["tx_feat_1"] = np.random.randn(len(tx_feats)).astype(np.float32)

        import time
        t0 = time.time()
        pyg_obj = networkx_to_pyg(graph, node_feats, tx_feats)
        pyg_obj = create_temporal_node_masks(pyg_obj, df_sample, pyg_obj.node_ids)
        elapsed = time.time() - t0

        print(f"PyG conversion completed in {elapsed:.3f} seconds.")
        print(f"PyG Data Summary: {pyg_obj}")
        print(f"Node Features (x): {pyg_obj.x.shape}")
        print(f"Edge Index: {pyg_obj.edge_index.shape}")
        print(f"Edge Attr: {pyg_obj.edge_attr.shape}")
        print(f"Train nodes: {pyg_obj.train_mask.sum().item():,}")
        print(f"Val nodes:   {pyg_obj.val_mask.sum().item():,}")
        print(f"Test nodes:  {pyg_obj.test_mask.sum().item():,}")
        print("All PyG assertions passed successfully!")
    else:
        print(f"Sample data not found at {DATA_PATH}.")
