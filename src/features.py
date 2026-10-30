"""
Feature engineering module for Abuse Ring Sentinel.

Constructs multi-level AML features across three distinct granularities:
    1. Sender-Side Graph Features: PageRank, HITS hub scores, degree ratios, and flow asymmetry.
    2. Account / Node Features: Transaction volume, incoming/outgoing flow imbalances, and counterparties.
    3. Transaction-Level Features: Cyclic temporal embeddings, velocity, structuring/smurfing signals,
       and sender graph metric merges.
"""

from __future__ import annotations

from typing import Tuple
import networkx as nx
import numpy as np
import pandas as pd

EPS = 1e-9

REQUIRED_TRANSACTION_COLUMNS: list[str] = [
    "nameOrig",
    "nameDest",
    "amount",
    "timestamp",
    "type",
    "payment_method",
]

REQUIRED_SENDER_GRAPH_COLUMNS: list[str] = [
    "account_id",
    "sender_pagerank",
    "sender_hub_score",
    "sender_indegree",
    "sender_outdegree",
    "sender_in_out_ratio",
    "sender_flow_asymmetry",
]


def build_sender_graph_features(
    graph: nx.MultiDiGraph | nx.DiGraph,
) -> pd.DataFrame:
    """
    Compute structural centrality, authority, and flow features for each account node.

    Features:
        - sender_pagerank: Steady-state random walk probability (structural influence).
        - sender_hub_score: HITS hub authority score (identifies distribution hubs).
        - sender_indegree & sender_outdegree: Unique transaction counts in and out.
        - sender_in_out_ratio: Ratio of received vs sent transfers.
        - sender_flow_asymmetry: Absolute degree difference normalized by total volume (mule flag).

    Parameters
    ----------
    graph : nx.MultiDiGraph or nx.DiGraph
        Master transaction network.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching REQUIRED_SENDER_GRAPH_COLUMNS.
    """
    if not isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        raise TypeError("graph must be a NetworkX DiGraph or MultiDiGraph.")

    nodes = list(graph.nodes)

    # 1. PageRank centrality
    try:
        pagerank = nx.pagerank(graph, alpha=0.85)
    except nx.NetworkXError:
        pagerank = {node: 0.0 for node in nodes}

    # 2. HITS hub scores
    try:
        hubs, _ = nx.hits(graph, max_iter=1000, normalized=True)
    except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
        hubs = {node: 0.0 for node in nodes}

    # 3. In-degree and Out-degree
    indegree = dict(graph.in_degree())
    outdegree = dict(graph.out_degree())

    # 4. Vectorized feature array construction
    in_arr = np.array([indegree.get(n, 0) for n in nodes], dtype=np.float32)
    out_arr = np.array([outdegree.get(n, 0) for n in nodes], dtype=np.float32)
    pr_arr = np.array([pagerank.get(n, 0.0) for n in nodes], dtype=np.float32)
    hub_arr = np.array([hubs.get(n, 0.0) for n in nodes], dtype=np.float32)

    in_out_ratio = in_arr / (out_arr + EPS)
    flow_asymmetry = np.abs(out_arr - in_arr) / (out_arr + in_arr + EPS)

    df_sender = pd.DataFrame({
        "account_id": nodes,
        "sender_pagerank": pr_arr,
        "sender_hub_score": hub_arr,
        "sender_indegree": in_arr,
        "sender_outdegree": out_arr,
        "sender_in_out_ratio": in_out_ratio,
        "sender_flow_asymmetry": flow_asymmetry,
    })

    return df_sender.reset_index(drop=True)


def build_account_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute aggregated historical transaction and flow statistics for all accounts.

    Captures entity-level volume, net flow, counterparty diversity, and historical fraud labels.

    Parameters
    ----------
    df : pd.DataFrame
        Transactions containing 'nameOrig', 'nameDest', 'amount', and optionally 'isFraud'.

    Returns
    -------
    pd.DataFrame
        Account features indexed by canonical account_id.
    """
    nodes = pd.Index(
        pd.unique(
            pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True)
        ),
        name="account_id",
    )

    # 1. Outgoing transfer behavior (as sender)
    outgoing = (
        df.groupby("nameOrig")["amount"]
        .agg(
            outgoing_count="size",
            outgoing_total="sum",
            outgoing_mean="mean",
            outgoing_max="max",
            outgoing_min="min",
        )
    )

    # 2. Incoming transfer behavior (as receiver)
    incoming = (
        df.groupby("nameDest")["amount"]
        .agg(
            incoming_count="size",
            incoming_total="sum",
            incoming_mean="mean",
            incoming_max="max",
            incoming_min="min",
        )
    )

    # 3. Join on canonical node index
    node_features = (
        pd.DataFrame(index=nodes)
        .join(outgoing)
        .join(incoming)
        .fillna(0.0)
    )

    # 4. Net volume and flow ratio metrics
    node_features["total_transaction_count"] = (
        node_features["outgoing_count"] + node_features["incoming_count"]
    )
    node_features["total_amount"] = (
        node_features["outgoing_total"] + node_features["incoming_total"]
    )
    node_features["net_flow"] = (
        node_features["incoming_total"] - node_features["outgoing_total"]
    )
    node_features["outflow_ratio"] = (
        node_features["outgoing_total"]
        / (node_features["incoming_total"] + node_features["outgoing_total"] + EPS)
    )
    node_features["flow_imbalance"] = (
        np.abs(node_features["net_flow"])
        / (node_features["incoming_total"] + node_features["outgoing_total"] + EPS)
    )

    # 5. Counterparty diversity
    node_features["unique_senders"] = (
        df.groupby("nameDest")["nameOrig"].nunique().reindex(nodes).fillna(0.0)
    )
    node_features["unique_recipients"] = (
        df.groupby("nameOrig")["nameDest"].nunique().reindex(nodes).fillna(0.0)
    )

    # 6. Amount distribution across all interactions
    node_features["mean_transaction_amount"] = (
        node_features["total_amount"] / (node_features["total_transaction_count"] + EPS)
    )
    node_features["max_transaction_amount"] = (
        node_features[["outgoing_max", "incoming_max"]].max(axis=1)
    )

    # For minimum amount, ignore 0.0 values from accounts with zero incoming or outgoing
    mins = pd.concat([
        node_features["outgoing_min"].replace(0.0, np.nan),
        node_features["incoming_min"].replace(0.0, np.nan),
    ], axis=1)
    node_features["min_transaction_amount"] = mins.min(axis=1).fillna(0.0)

    # 7. Ground truth node label (1 if account originated at least one fraudulent transfer)
    if "isFraud" in df.columns:
        node_features["label"] = (
            df.groupby("nameOrig")["isFraud"].max().reindex(nodes).fillna(0).astype(np.int8)
        )
    else:
        node_features["label"] = np.int8(0)

    # Clean any infinite or NaN values
    num_cols = node_features.select_dtypes(include=np.number).columns
    node_features[num_cols] = (
        node_features[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    return node_features.reset_index()


def build_transaction_features(
    df: pd.DataFrame,
    sender_graph_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct tabular transaction-level features for the XGBoost risk model.

    Features include:
        - Normalized & log-transformed amounts.
        - One-hot encoded transaction types and payment methods.
        - Cyclical 24-hour and 7-day trigonometric embeddings.
        - Sender/receiver temporal velocity (rapid activity bursts).
        - Structuring / smurfing signal (similar amounts sent in rapid succession).
        - Merged sender-side structural network metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized transactions with required metadata.
    sender_graph_features : pd.DataFrame
        Precomputed structural metrics from build_sender_graph_features().

    Returns
    -------
    pd.DataFrame
        Tabular feature matrix ready for XGBoost modeling.
    """
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for transaction features: {missing}")

    missing_graph = [
        c for c in REQUIRED_SENDER_GRAPH_COLUMNS if c not in sender_graph_features.columns
    ]
    if missing_graph:
        raise ValueError(f"Missing required sender graph columns: {missing_graph}")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")

    # Ensure chronological ordering for velocity calculations
    work = (
        work.dropna(subset=["nameOrig", "nameDest", "amount", "timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # 1. Base Identifiers and Amount Features
    tx_features = pd.DataFrame(index=work.index)
    tx_features["transaction_id"] = work["transaction_id"]
    tx_features["nameOrig"] = work["nameOrig"]
    tx_features["nameDest"] = work["nameDest"]
    tx_features["timestamp"] = work["timestamp"]
    tx_features["amount"] = work["amount"]
    tx_features["log_amount"] = np.log1p(work["amount"].clip(lower=0.0))

    # 2. One-hot categorical encodings
    type_dummies = pd.get_dummies(work["type"].astype(str), prefix="type", dtype=np.float32)
    payment_dummies = pd.get_dummies(work["payment_method"].astype(str), prefix="payment", dtype=np.float32)
    tx_features = pd.concat([tx_features, type_dummies, payment_dummies], axis=1)

    # 3. Continuous Cyclical Temporal Embeddings
    hour_decimal = work["timestamp"].dt.hour + work["timestamp"].dt.minute / 60.0
    day_of_week = work["timestamp"].dt.dayofweek

    tx_features["hour_sin"] = np.sin(2 * np.pi * hour_decimal / 24.0).astype(np.float32)
    tx_features["hour_cos"] = np.cos(2 * np.pi * hour_decimal / 24.0).astype(np.float32)
    tx_features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0).astype(np.float32)
    tx_features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0).astype(np.float32)
    tx_features["day_of_month"] = work["timestamp"].dt.day.astype(np.int8)
    tx_features["month"] = work["timestamp"].dt.month.astype(np.int8)

    # 4. Temporal Velocity & Rapid Activity Bursts
    sender_prev_time = work.groupby("nameOrig")["timestamp"].shift(1)
    sender_prev_amt = work.groupby("nameOrig")["amount"].shift(1)
    sender_time_diff = (work["timestamp"] - sender_prev_time).dt.total_seconds()

    receiver_prev_time = work.groupby("nameDest")["timestamp"].shift(1)
    receiver_time_diff = (work["timestamp"] - receiver_prev_time).dt.total_seconds()

    # Rapid activity flag: subsequent transfer within 5 minutes (300 seconds)
    tx_features["rapid_sender_activity"] = (
        (sender_time_diff >= 0) & (sender_time_diff <= 300)
    ).astype(np.int8)
    tx_features["rapid_receiver_activity"] = (
        (receiver_time_diff >= 0) & (receiver_time_diff <= 300)
    ).astype(np.int8)

    # 5. Structuring / Smurfing Detection
    # Identifies consecutive transactions within 1 hour where amounts are >= 90% identical
    amt_diff = (work["amount"] - sender_prev_amt).abs()
    amt_max = np.maximum(work["amount"], sender_prev_amt) + EPS
    similarity = (1.0 - amt_diff / amt_max).clip(0.0, 1.0).fillna(0.0)

    tx_features["amount_similarity_prev"] = similarity.astype(np.float32)
    tx_features["structuring_signal"] = (
        (sender_time_diff >= 0) & (sender_time_diff <= 3600) & (similarity >= 0.90)
    ).astype(np.int8)

    # 6. Merge Sender Graph Metrics (authorities, PageRank, degree asymmetry)
    tx_features = tx_features.merge(
        sender_graph_features[REQUIRED_SENDER_GRAPH_COLUMNS],
        how="left",
        left_on="nameOrig",
        right_on="account_id",
        validate="many_to_one",
    ).drop(columns=["account_id"])

    # 7. Labels
    tx_features["isFraud"] = work["isFraud"].astype(np.int8)
    tx_features["laundering_typology"] = work["laundering_typology"]

    return tx_features.reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path
    import time
    from ingest import load_transactions
    from graph_builder import build_graph

    DATA_PATH = Path("data/raw/amlnet/AMLNet_v2_transactions.csv")
    if DATA_PATH.exists():
        print("Testing features.py on 5,000 transaction sample...")
        df_sample = load_transactions(DATA_PATH, nrows=5000)
        graph, _, _ = build_graph(df_sample)

        t0 = time.time()
        sender_gf = build_sender_graph_features(graph)
        account_f = build_account_features(df_sample)
        tx_f = build_transaction_features(df_sample, sender_gf)
        elapsed = time.time() - t0

        print(f"All features engineered in {elapsed:.3f} seconds.")
        print(f"Sender graph features: {sender_gf.shape}")
        print(f"Account features:      {account_f.shape}")
        print(f"Transaction features:  {tx_f.shape}")
        print(f"Structuring signals detected: {tx_f['structuring_signal'].sum():,}")
        print("Feature engineering verification passed successfully!")
    else:
        print(f"Data file not found at {DATA_PATH}.")
