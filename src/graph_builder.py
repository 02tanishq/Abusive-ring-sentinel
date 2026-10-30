"""
Master graph construction and feature integration for Abuse Ring Sentinel.

Constructs a canonical NetworkX MultiDiGraph representing the entire financial
transaction network:
    - Nodes: Accounts / Entities (nameOrig, nameDest)
    - Edges: Individual transactions keyed by stable integer `transaction_id`
    - Node / Edge Attributes: Tabular & structural features for downstream GNN & topology mining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import networkx as nx
import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "transaction_id",
    "nameOrig",
    "nameDest",
]


def build_graph(
    df: pd.DataFrame,
) -> Tuple[nx.MultiDiGraph, dict[str, int], dict[int, str]]:
    """
    Build the canonical NetworkX MultiDiGraph from normalized transactions.

    A MultiDiGraph is used because financial actors can transact multiple times
    in the same direction. Each transaction is preserved as a unique directed edge
    keyed by its integer `transaction_id`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at minimum: 'transaction_id', 'nameOrig', 'nameDest'.

    Returns
    -------
    graph : nx.MultiDiGraph
        The constructed master MultiDiGraph.
    node_to_idx : dict[str, int]
        Deterministic mapping from account_id to 0-based node index for GNN tensor alignment.
    idx_to_node : dict[int, str]
        Reverse mapping from 0-based index to account_id.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for graph construction: {missing}")

    # 1. Deterministic canonical node ordering
    # We preserve unique accounts from both origin and destination pools
    nodes = pd.Index(
        pd.unique(
            pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
        ),
        name="account_id",
    )

    node_to_idx: dict[str, int] = {str(node): idx for idx, node in enumerate(nodes)}
    idx_to_node: dict[int, str] = {idx: str(node) for idx, node in enumerate(nodes)}

    # 2. Initialize MultiDiGraph and add nodes in canonical order
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes)

    # Cache canonical mappings in the graph metadata dictionary
    graph.graph["node_to_idx"] = node_to_idx
    graph.graph["idx_to_node"] = idx_to_node

    # 3. Add edges in batch using a generator for high throughput and low memory overhead
    edge_batch = (
        (
            row.nameOrig,
            row.nameDest,
            int(row.transaction_id),
            {"transaction_id": int(row.transaction_id)},
        )
        for row in df[["transaction_id", "nameOrig", "nameDest"]].itertuples(index=False)
    )
    graph.add_edges_from(edge_batch)

    # Verify node ordering invariant for PyG / GNN alignment
    assert list(graph.nodes) == list(nodes), (
        "Graph node ordering diverged from canonical index."
    )

    return graph, node_to_idx, idx_to_node


def integrate_features(
    graph: nx.MultiDiGraph,
    node_features: pd.DataFrame,
    transaction_features: pd.DataFrame | None = None,
) -> nx.MultiDiGraph:
    """
    Integrate account-level and transaction-level features into the master graph.

    Uses vectorized NetworkX attribute mapping to avoid slow Python dictionary
    lookups over millions of edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The master transaction graph.
    node_features : pd.DataFrame
        Account-level features indexed or containing 'account_id'.
    transaction_features : pd.DataFrame, optional
        Transaction-level features containing 'transaction_id'.

    Returns
    -------
    nx.MultiDiGraph
        The updated master graph with attached node and edge attributes.
    """
    # 1. Validate & attach node features
    if "account_id" not in node_features.columns:
        raise ValueError("node_features must contain an 'account_id' column.")

    node_cols = [c for c in node_features.columns if c != "account_id"]
    node_dict = (
        node_features.set_index("account_id")[node_cols]
        .to_dict(orient="index")
    )
    nx.set_node_attributes(graph, node_dict)

    # 2. Validate & attach transaction / edge features if provided
    if transaction_features is not None:
        if "transaction_id" not in transaction_features.columns:
            raise ValueError("transaction_features must contain 'transaction_id'.")

        tx_cols = [
            c for c in transaction_features.columns
            if c not in {"transaction_id", "nameOrig", "nameDest"}
        ]

        # Use multi-index (u, v, key) for fast bulk edge attribute assignment
        edge_dict = (
            transaction_features
            .set_index(["nameOrig", "nameDest", "transaction_id"])[tx_cols]
            .to_dict(orient="index")
        )
        nx.set_edge_attributes(graph, edge_dict)

    return graph


if __name__ == "__main__":
    import time
    from ingest import load_transactions

    DATA_PATH = Path("data/raw/amlnet/AMLNet_v2_transactions.csv")
    if DATA_PATH.exists():
        print("Testing graph_builder.py on 5,000 transaction sample...")
        df_sample = load_transactions(DATA_PATH, nrows=5000)

        t0 = time.time()
        g, n2i, i2n = build_graph(df_sample)
        elapsed = time.time() - t0

        print(f"Graph built in {elapsed:.3f} seconds.")
        print(f"Nodes: {g.number_of_nodes():,}")
        print(f"Edges: {g.number_of_edges():,}")
        print(f"Node to index sample: {list(n2i.items())[:3]}")

        # Quick verification of edge key preservation
        sample_edge = next(iter(g.edges(keys=True, data=True)))
        print(f"Sample Edge (u, v, key, data): {sample_edge}")
        assert "transaction_id" in sample_edge[3], "Edge missing transaction_id attribute!"
        print("All structural checks passed!")
    else:
        print(f"Data file not found at {DATA_PATH}. Place dataset in data/raw/ to test.")
