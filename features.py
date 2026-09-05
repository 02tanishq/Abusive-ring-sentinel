# src/features.py

import numpy as np
import pandas as pd
import networkx as nx


EPS = 1e-9


def build_transaction_features(
    df: pd.DataFrame,
    sender_graph_features: pd.DataFrame,
) -> pd.DataFrame:
    
    required_columns = [
        "nameOrig",
        "nameDest",
        "amount",
        "timestamp",
        "type",
        "payment_method",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Work only with columns needed by this feature pipeline.
    # We do NOT return a copy of the entire original DataFrame.
    # --------------------------------------------------------

    columns = [
        "transaction_id",
        "nameOrig",
        "nameDest",
        "amount",
        "timestamp",
        "type",
        "payment_method",
        "isFraud",
        "isMoneyLaundering",
        "laundering_typology",
    ]

    # transaction_id may not exist in the raw CSV, because
    # ingest.py currently does not create it.

    work = df[columns].copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    work["amount"] = pd.to_numeric(
        work["amount"],
        errors="coerce",
    )

    work = (
        work
        .dropna(
            subset=[
                "nameOrig",
                "nameDest",
                "amount",
                "timestamp",
            ]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Start NEW feature DataFrame

    transaction_features = pd.DataFrame(index=work.index)

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    transaction_features["transaction_id"] = work["transaction_id"]
    transaction_features["nameOrig"] = work["nameOrig"]
    transaction_features["nameDest"] = work["nameDest"]
    transaction_features["timestamp"] = work["timestamp"]

    # 1. Transaction transaction_features

    transaction_features["amount"] = work["amount"]

    transaction_features["log_amount"] = np.log1p(
        work["amount"].clip(lower=0)
    )

    # Transaction type
    type_features = pd.get_dummies(
        work["type"].astype(str),
        prefix="type",
        dtype=np.float32,
    )

    transaction_features = pd.concat(
        [transaction_features, type_features],
        axis=1,
    )

    # Payment method
    payment_features = pd.get_dummies(
        work["payment_method"].astype(str),
        prefix="payment",
        dtype=np.float32,
    )

    transaction_features = pd.concat(
        [transaction_features, payment_features],
        axis=1,
    )

    # 2. Cyclic temporal transaction_features

    hour = work["timestamp"].dt.hour
    minute = work["timestamp"].dt.minute

    hour_decimal = hour + minute / 60.0

    transaction_features["hour_sin"] = np.sin(
        2 * np.pi * hour_decimal / 24.0
    )

    transaction_features["hour_cos"] = np.cos(
        2 * np.pi * hour_decimal / 24.0
    )

    day_of_week = work["timestamp"].dt.dayofweek

    transaction_features["dow_sin"] = np.sin(
        2 * np.pi * day_of_week / 7.0
    )

    transaction_features["dow_cos"] = np.cos(
        2 * np.pi * day_of_week / 7.0
    )

    transaction_features["day_of_month"] = (
        work["timestamp"].dt.day
    )

    transaction_features["month"] = (
        work["timestamp"].dt.month
    )

    # 3. Sender temporal transaction_features

    sender_previous_time = (
        work.groupby("nameOrig")["timestamp"]
        .shift(1)
    )

    sender_previous_amount = (
        work.groupby("nameOrig")["amount"]
        .shift(1)
    )

    sender_time_since_previous = (
        work["timestamp"] - sender_previous_time
    ).dt.total_seconds()

    # 4. Receiver temporal transaction_features

    receiver_previous_time = (
        work.groupby("nameDest")["timestamp"]
        .shift(1)
    )

    receiver_time_since_previous = (
        work["timestamp"] - receiver_previous_time
    ).dt.total_seconds()

    # 5. Rapid activity

    transaction_features["rapid_sender_activity"] = (
        (
            sender_time_since_previous >= 0
        )
        & (
            sender_time_since_previous <= 300
        )
    ).astype(np.int8)

    transaction_features["rapid_receiver_activity"] = (
        (
            receiver_time_since_previous >= 0
        )
        & (
            receiver_time_since_previous <= 300
        )
    ).astype(np.int8)

    # 6. Structuring transaction_features

    amount_similarity_previous = (
        1
        - (
            (
                work["amount"]
                - sender_previous_amount
            ).abs()
            / (
                np.maximum(
                    work["amount"],
                    sender_previous_amount,
                )
                + EPS
            )
        )
    )

    transaction_features["amount_similarity_prev"] = (
        amount_similarity_previous
        .clip(0, 1)
        .fillna(0)
    )

    structuring_window_seconds = 3600
    similarity_threshold = 0.90

    transaction_features["structuring_signal"] = (
        (
            sender_time_since_previous >= 0
        )
        & (
            sender_time_since_previous
            <= structuring_window_seconds
        )
        & (
            transaction_features["amount_similarity_prev"]
            >= similarity_threshold
        )
    ).astype(np.int8)

    # 7. Sender-side graph features
    #
    # These features were already computed from the master
    # NetworkX graph in pipeline.py. We only merge them here.

    required_graph_columns = [
        "account_id",
        "sender_pagerank",
        "sender_hub_score",
        "sender_indegree",
        "sender_outdegree",
        "sender_in_out_ratio",
        "sender_flow_asymmetry",
    ]

    missing_graph_columns = [
        column
        for column in required_graph_columns
        if column not in sender_graph_features.columns
    ]

    if missing_graph_columns:
        raise ValueError(
            f"Missing sender graph features: {missing_graph_columns}"
        )

    transaction_features = transaction_features.merge(
        sender_graph_features[
            required_graph_columns
        ],
        how="left",
        left_on="nameOrig",
        right_on="account_id",
        validate="many_to_one",
    )

    transaction_features = transaction_features.drop(
        columns=["account_id"]
    )

    # 8. Labels

    transaction_features["isFraud"] = (
        work["isFraud"]
        .astype(np.int8)
    )

    transaction_features["laundering_typology"] = (
        work["laundering_typology"]
    )

    return transaction_features.reset_index(drop=True)

# ============================================================
# Sender-side graph transaction_features
# ============================================================

def build_sender_graph_features(
    graph: nx.MultiDiGraph,
) -> pd.DataFrame:
    if not isinstance(
        graph,
        (nx.DiGraph, nx.MultiDiGraph),
    ):
        raise TypeError(
            "graph must be a NetworkX DiGraph or MultiDiGraph."
        )

    # 1. PageRank

    try:
        pagerank = nx.pagerank(
            graph,
            alpha=0.85,
        )
    except nx.NetworkXError:
        pagerank = {
            node: 0.0
            for node in graph.nodes
        }

    # 2. Hub score

    try:
        hubs, _ = nx.hits(
            graph,
            max_iter=1000,
            normalized=True,
        )
    except (
        nx.NetworkXError,
        nx.PowerIterationFailedConvergence,
    ):
        hubs = {
            node: 0.0
            for node in graph.nodes
        }

    # 3. Degree

    indegree = dict(
        graph.in_degree()
    )

    outdegree = dict(
        graph.out_degree()
    )

    # 5. Construct feature table
 

    rows = []

    for node in graph.nodes:

        in_degree = indegree.get(
            node,
            0,
        )

        out_degree = outdegree.get(
            node,
            0,
        )

        inbound_outbound_ratio = (
            in_degree
            / (
                out_degree
                + EPS
            )
        )

        flow_asymmetry = (
            abs(
                out_degree
                - in_degree
            )
            / (
                out_degree
                + in_degree
                + EPS
            )
        )

        rows.append(
            {
                "account_id": node,

                "sender_pagerank": pagerank.get(
                    node,
                    0.0,
                ),

                "sender_hub_score": hubs.get(
                    node,
                    0.0,
                ),

                "sender_indegree": in_degree,

                "sender_outdegree": out_degree,

                "sender_in_out_ratio": (
                    inbound_outbound_ratio
                ),

                "sender_flow_asymmetry": (
                    flow_asymmetry
                ),

            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Account / node transaction_features
# ============================================================

def build_account_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per account."""

    nodes = pd.Index(
        pd.unique(
            pd.concat(
                [
                    df["nameOrig"],
                    df["nameDest"],
                ],
                ignore_index=True,
            )
        ),
        name="account_id",
    )

    node_features = pd.DataFrame(
        index=nodes
    )

    
    # Incoming / outgoing behavior
    

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

    node_features = node_features.join(
        outgoing
    )

    node_features = node_features.join(
        incoming
    )

    node_features = node_features.fillna(0)

    node_features["total_transaction_count"] = (
        node_features["outgoing_count"]
        + node_features["incoming_count"]
    )

    node_features["total_amount"] = (
        node_features["outgoing_total"]
        + node_features["incoming_total"]
    )

    
    # Flow behavior
    

    node_features["net_flow"] = (
        node_features["incoming_total"]
        - node_features["outgoing_total"]
    )

    # Bounded 0-1 ratio.
    node_features["outflow_ratio"] = (
        node_features["outgoing_total"]
        / (
            node_features["incoming_total"]
            + node_features["outgoing_total"]
            + EPS
        )
    )

    node_features["flow_imbalance"] = (
        abs(node_features["net_flow"])
        / (
            node_features["incoming_total"]
            + node_features["outgoing_total"]
            + EPS
        )
    )

    
    # Counterparties
    

    node_features["unique_senders"] = (
        df.groupby("nameDest")["nameOrig"]
        .nunique()
        .reindex(node_features.index)
        .fillna(0)
    )

    node_features["unique_recipients"] = (
        df.groupby("nameOrig")["nameDest"]
        .nunique()
        .reindex(node_features.index)
        .fillna(0)
    )
    
    # Amount behavior
    

    node_features["mean_transaction_amount"] = (
        node_features["total_amount"]
        / (
            node_features[
                "total_transaction_count"
            ]
            + EPS
        )
    )

    node_features["max_transaction_amount"] = (
        pd.concat(
            [
                node_features["outgoing_max"],
                node_features["incoming_max"],
            ],
            axis=1,
        ).max(axis=1)
    )

    mins = pd.concat(
        [
            node_features[
                "outgoing_min"
            ].replace(0, np.nan),

            node_features[
                "incoming_min"
            ].replace(0, np.nan),
        ],
        axis=1,
    )

    node_features["min_transaction_amount"] = (
        mins.min(axis=1)
        .fillna(0)
    )

    # label
    node_features["label"] = (
    df.groupby("nameOrig")["isFraud"]
    .max()
    .reindex(node_features.index)
    .fillna(0)
    .astype(np.int8)
     )
    
    # Clean numeric columns
    

    numeric_columns = (
        node_features
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    node_features[
        numeric_columns
    ] = (
        node_features[
            numeric_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    node_features.index.name = (
        "account_id"
    )

    return node_features.reset_index()



