"""
Deep Risk Analysis
------------------

Analyzes each topology-generated candidate independently.

Architecture:
    Master Graph
        |
        +--> topology candidate accounts
        |
        +--> transaction_id
                 |
                 v
        clean_transactions
                 |
                 +--> isFraud (ground truth)
                 |
                 +--> transaction data

Risk inputs:
    - node_risk_results.parquet
    - transaction_risk_all.parquet

Important:
    - No candidate merging.
    - No new subgraph generation.
    - No graph rebuilding.
    - Each topology candidate is analyzed independently.
    - Master graph remains structural only.
    - Ground truth comes from clean_transactions via transaction_id.
"""

from __future__ import annotations

import ast
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS_DIR = Path("artifacts")

MASTER_GRAPH_PATH = ARTIFACTS_DIR / "master_graph.pkl"
CLEAN_TX_PATH = ARTIFACTS_DIR / "clean_transactions.parquet"
NODE_RISK_PATH = ARTIFACTS_DIR / "node_risk_results.parquet"
TX_RISK_PATH = ARTIFACTS_DIR / "transaction_risk_all.parquet"

# Change this to the exact output of topology.py if required.
TOPOLOGY_CANDIDATE_PATHS = [
    ARTIFACTS_DIR / "topology_patterns.parquet",
    ARTIFACTS_DIR / "topology_results.parquet",
    ARTIFACTS_DIR / "topology_candidates.parquet",
    ARTIFACTS_DIR / "pattern_results.parquet",
    ARTIFACTS_DIR / "pattern_candidates.parquet",
]

OUTPUT_PATH = ARTIFACTS_DIR / "ring_analysis.parquet"

ALERT_THRESHOLD = 0.50
FALSE_POSITIVE_COST = 5000.0

EPS = 1e-9


# ============================================================
# GENERAL HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _normalize_id(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):

        if (
            math.isfinite(float(value))
            and float(value).is_integer()
        ):
            return str(int(value))

    return str(value).strip()


def _first_existing(paths: Iterable[Path]) -> Path:

    for path in paths:

        if path.exists():
            return path

    paths_text = "\n".join(
        f"    {path}"
        for path in paths
    )

    raise FileNotFoundError(
        "Could not find topology candidate file.\n"
        "Checked:\n"
        f"{paths_text}"
    )


# ============================================================
# LOAD MASTER GRAPH
# ============================================================

def _load_master_graph(
    path: Path,
) -> nx.MultiDiGraph:

    with path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, tuple):
        graph = obj[0]
    else:
        graph = obj

    if not isinstance(graph, nx.MultiDiGraph):

        raise TypeError(
            "master_graph.pkl must contain nx.MultiDiGraph. "
            f"Found: {type(graph).__name__}"
        )

    return graph


# ============================================================
# ACCOUNT PARSING
# ============================================================

def _parse_accounts(value: Any) -> list[str]:

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    # Already list-like
    if isinstance(
        value,
        (
            list,
            tuple,
            set,
            np.ndarray,
            pd.Series,
        ),
    ):

        values = list(value)

    else:

        text = str(value).strip()

        values = None

        # Try JSON / Python literal.
        if text.startswith(
            (
                "[",
                "(",
                "{",
            )
        ):

            for parser in (
                json.loads,
                ast.literal_eval,
            ):

                try:

                    parsed = parser(text)

                    if isinstance(
                        parsed,
                        dict,
                    ):

                        for key in (
                            "accounts",
                            "account_ids",
                            "nodes",
                            "members",
                            "candidate_accounts",
                        ):

                            if key in parsed:

                                parsed = parsed[key]
                                break

                    if isinstance(
                        parsed,
                        (
                            list,
                            tuple,
                            set,
                        ),
                    ):

                        values = list(parsed)
                        break

                except Exception:
                    pass

        if values is None:

            if "|" in text:

                values = text.split("|")

            elif "," in text:

                values = text.split(",")

            elif ";" in text:

                values = text.split(";")

            else:

                values = [text]

    output = []
    seen = set()

    for value in values:

        account = _normalize_id(value)

        account = account.strip("'\"")

        if not account:
            continue

        if account not in seen:

            seen.add(account)
            output.append(account)

    return output


def _find_account_column(
    df: pd.DataFrame,
) -> str | None:

    candidate_columns = [
        "accounts",
        "account_ids",
        "candidate_accounts",
        "members",
        "nodes",
        "subgraph_nodes",
        "ring_accounts",
        "member_accounts",
    ]

    for column in candidate_columns:

        if column in df.columns:
            return column

    # Fallback:
    # Look for a column that appears to contain multiple account IDs.
    excluded = {
        "candidate_id",
        "pattern_type",
        "pattern",
        "num_accounts",
        "num_members",
        "num_transactions",
        "ring_score",
    }

    for column in df.columns:

        if column in excluded:
            continue

        sample = (
            df[column]
            .dropna()
            .head(20)
        )

        if sample.empty:
            continue

        valid = 0

        for value in sample:

            accounts = _parse_accounts(value)

            if len(accounts) >= 2:
                valid += 1

        if valid >= 2:
            return column

    return None


# ============================================================
# TOPOLOGY NORMALIZATION
# ============================================================

def _canonical_pattern_type(
    row: pd.Series,
) -> str:

    raw = str(
        row.get(
            "pattern_type",
            row.get(
                "pattern",
                "",
            ),
        )
    ).strip().lower()

    # Normalize cycle naming.
    if raw in {
        "cycle",
        "cycles",
        "simple_cycle",
        "simple_cycles",
    }:

        length = row.get(
            "num_members",
            row.get(
                "cycle_length",
                row.get(
                    "num_accounts",
                    0,
                ),
            ),
        )

        try:

            return f"{int(length)}_cycle"

        except Exception:

            return "cycle"

    raw = (
        raw
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {

        "fanout": "fan_out",

        "fanout_pattern": "fan_out",

        "fan_in_pattern": "fan_in",

        "reciprocal_pair": "reciprocal",

        "reciprocal_pairs": "reciprocal",
    }

    return aliases.get(
        raw,
        raw,
    )


# ============================================================
# TRANSACTION LOOKUP
# ============================================================

def _build_transaction_lookup(
    clean_tx: pd.DataFrame,
) -> dict[int, dict[str, Any]]:

    required_columns = {
        "transaction_id",
        "nameOrig",
        "nameDest",
        "amount",
        "timestamp",
        "isFraud",
    }

    missing = (
        required_columns
        - set(clean_tx.columns)
    )

    if missing:

        raise ValueError(
            "clean_transactions.parquet is missing "
            f"columns: {sorted(missing)}"
        )

    columns = [
        "transaction_id",
        "nameOrig",
        "nameDest",
        "amount",
        "timestamp",
        "type",
        "payment_method",
        "isFraud",
    ]

    columns = [
        column
        for column in columns
        if column in clean_tx.columns
    ]

    data = clean_tx[columns].copy()

    data["transaction_id"] = (
        pd.to_numeric(
            data["transaction_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    data = data.dropna(
        subset=["transaction_id"]
    )

    data["transaction_id"] = (
        data["transaction_id"]
        .astype("int64")
    )

    data["nameOrig"] = (
        data["nameOrig"]
        .astype(str)
    )

    data["nameDest"] = (
        data["nameDest"]
        .astype(str)
    )

    data["amount"] = (
        pd.to_numeric(
            data["amount"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data["isFraud"] = (
        pd.to_numeric(
            data["isFraud"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )

    # There should be one row per transaction_id.
    data = data.drop_duplicates(
        subset=["transaction_id"],
        keep="last",
    )

    return (
        data
        .set_index("transaction_id")
        .to_dict("index")
    )


# ============================================================
# RECOVER TRANSACTIONS FOR ONE TOPOLOGY CANDIDATE
# ============================================================

def recover_candidate_transactions(
    graph: nx.MultiDiGraph,
    candidate_accounts: list[str],
    transaction_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:

    """
    Recover only original transactions where:

        sender ∈ candidate_accounts
        AND
        receiver ∈ candidate_accounts

    No new graph is constructed.
    No candidates are merged.
    """

    account_set = set(
        candidate_accounts
    )

    rows = []

    seen_transaction_ids = set()

    for source in account_set:

        if source not in graph:
            continue

        for target in graph.successors(source):

            if target not in account_set:
                continue

            edge_data = graph.get_edge_data(
                source,
                target,
            )

            if not edge_data:
                continue

            for _, attributes in edge_data.items():

                transaction_id = attributes.get(
                    "transaction_id"
                )

                if transaction_id is None:
                    continue

                transaction_id = int(
                    transaction_id
                )

                if transaction_id in seen_transaction_ids:
                    continue

                transaction = transaction_lookup.get(
                    transaction_id
                )

                if transaction is None:
                    continue

                row = dict(transaction)

                row["transaction_id"] = (
                    transaction_id
                )

                rows.append(row)

                seen_transaction_ids.add(
                    transaction_id
                )

    if not rows:

        return pd.DataFrame(
            columns=[
                "transaction_id",
                "nameOrig",
                "nameDest",
                "amount",
                "timestamp",
                "isFraud",
            ]
        )

    result = pd.DataFrame(rows)

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result["amount"] = (
        pd.to_numeric(
            result["amount"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    result["isFraud"] = (
        pd.to_numeric(
            result["isFraud"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int8")
    )

    return (
        result
        .sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# NODE RISK
# ============================================================

def node_risk_components(
    candidate_accounts: list[str],
    node_risk: dict[str, float],
):

    if not candidate_accounts:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    risks = np.asarray(
        [
            _safe_float(
                node_risk.get(
                    account,
                    0.0,
                )
            )
            for account in candidate_accounts
        ],
        dtype=np.float64,
    )

    mean_risk = float(
        risks.mean()
    )

    max_risk = float(
        risks.max()
    )

    high_fraction = float(
        (risks >= 0.70).mean()
    )

    combined = (
        0.50 * mean_risk
        + 0.30 * max_risk
        + 0.20 * high_fraction
    )

    return (
        _clip01(mean_risk),
        _clip01(max_risk),
        _clip01(high_fraction),
        _clip01(combined),
    )


# ============================================================
# TRANSACTION RISK
# ============================================================

def transaction_risk_components(
    transaction_ids: Iterable[int],
    tx_risk: dict[int, float],
):

    transaction_ids = list(
        transaction_ids
    )

    if not transaction_ids:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    risks = np.asarray(
        [
            _safe_float(
                tx_risk.get(
                    int(transaction_id),
                    0.0,
                )
            )
            for transaction_id in transaction_ids
        ],
        dtype=np.float64,
    )

    mean_risk = float(
        risks.mean()
    )

    max_risk = float(
        risks.max()
    )

    high_fraction = float(
        (risks >= 0.70).mean()
    )

    combined = (
        0.40 * mean_risk
        + 0.40 * max_risk
        + 0.20 * high_fraction
    )

    return (
        _clip01(mean_risk),
        _clip01(max_risk),
        _clip01(high_fraction),
        _clip01(combined),
    )


# ============================================================
# TOPOLOGY STRENGTH
# ============================================================

def topology_strength(
    pattern_type: str,
    num_accounts: int,
) -> float:

    pattern_type = pattern_type.lower()

    cycle_strength = {

        "3_cycle": 1.00,

        "4_cycle": 0.95,

        "5_cycle": 0.90,

        "6_cycle": 0.85,
    }

    if pattern_type in cycle_strength:

        return cycle_strength[
            pattern_type
        ]

    if pattern_type == "reciprocal":

        return 0.30

    if pattern_type in {
        "fan_in",
        "fan_out",
    }:

        return _clip01(
            0.30
            + 0.20
            * min(
                num_accounts / 10.0,
                1.0,
            )
        )

    return 0.20


# ============================================================
# STRUCTURAL SIGNAL
# ============================================================

def structural_signal(
    tx: pd.DataFrame,
    candidate_accounts: list[str],
    pattern_type: str,
) -> float:

    if tx.empty:
        return 0.0

    pattern_type = pattern_type.lower()

    edges = set(
        zip(
            tx["nameOrig"].astype(str),
            tx["nameDest"].astype(str),
        )
    )

    # --------------------------------------------------------
    # Cycle
    # --------------------------------------------------------

    if pattern_type.endswith("_cycle"):

        numbers = re.findall(
            r"\d+",
            pattern_type,
        )

        if not numbers:
            return 0.0

        cycle_length = int(
            numbers[0]
        )

        candidate_set = set(
            candidate_accounts
        )

        H = nx.DiGraph()

        H.add_nodes_from(
            candidate_set
        )

        H.add_edges_from(
            edges
        )

        try:

            cyclic_nodes = set()

            cycles = nx.simple_cycles(
                H,
                length_bound=min(
                    cycle_length,
                    6,
                ),
            )

            for cycle in cycles:

                if len(cycle) == cycle_length:

                    cyclic_nodes.update(
                        cycle
                    )

            participation = (
                len(cyclic_nodes)
                / max(
                    len(candidate_set),
                    1,
                )
            )

        except Exception:

            participation = 0.0

        return _clip01(
            participation
        )

    # --------------------------------------------------------
    # Reciprocal
    # --------------------------------------------------------

    if pattern_type == "reciprocal":

        if not edges:
            return 0.0

        unordered_pairs = set()

        reciprocal_pairs = 0

        for source, target in edges:

            pair = tuple(
                sorted(
                    (
                        source,
                        target,
                    )
                )
            )

            if pair in unordered_pairs:
                continue

            unordered_pairs.add(pair)

            if (
                target,
                source,
            ) in edges:

                reciprocal_pairs += 1

        return _clip01(
            reciprocal_pairs
            / max(
                len(unordered_pairs),
                1,
            )
        )

    # --------------------------------------------------------
    # Fan-in / fan-out
    # --------------------------------------------------------

    if pattern_type in {
        "fan_in",
        "fan_out",
    }:

        if pattern_type == "fan_in":

            counts = (
                tx["nameDest"]
                .value_counts()
            )

        else:

            counts = (
                tx["nameOrig"]
                .value_counts()
            )

        if counts.empty:
            return 0.0

        max_count = float(
            counts.max()
        )

        total = float(
            counts.sum()
        )

        return _clip01(
            max_count
            / max(
                total,
                1.0,
            )
        )

    return 0.0


# ============================================================
# TEMPORAL SIGNAL
# ============================================================

def temporal_signal(
    tx: pd.DataFrame,
):

    if (
        tx.empty
        or tx["timestamp"].isna().all()
    ):

        return (
            0.0,
            0.0,
            0.0,
            np.nan,
        )

    timestamps = (
        tx["timestamp"]
        .dropna()
        .sort_values()
    )

    if len(timestamps) < 2:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    span_days = (
        (
            timestamps.max()
            - timestamps.min()
        ).total_seconds()
        / 86400.0
    )

    # Rapid activity:
    # <= 1 day
    rapid_score = (
        1.0
        if span_days <= 1.0
        else 0.0
    )

    # Layering:
    # exponentially decreasing beyond 1 day.
    layering_score = math.exp(
        -max(
            span_days - 1.0,
            0.0,
        )
        / 20.0
    )

    # Placement:
    # exponentially decreasing beyond 3 days.
    placement_score = math.exp(
        -max(
            span_days - 3.0,
            0.0,
        )
        / 20.0
    )

    temporal_score = (
        0.45 * rapid_score
        + 0.30 * layering_score
        + 0.25 * placement_score
    )

    return (
        _clip01(rapid_score),
        _clip01(layering_score),
        _clip01(temporal_score),
        float(span_days),
    )


# ============================================================
# AMOUNT SIGNAL
# ============================================================

def amount_signal(
    tx: pd.DataFrame,
):

    if tx.empty:

        return (
            0.0,
            0.0,
            0.0,
        )

    amounts = (
        pd.to_numeric(
            tx["amount"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    positive = amounts[
        amounts > 0
    ]

    # --------------------------------------------------------
    # Amount similarity
    # --------------------------------------------------------

    if len(positive) <= 1:

        similarity = 0.0

    else:

        mean_amount = float(
            positive.mean()
        )

        std_amount = float(
            positive.std(
                ddof=0
            )
        )

        coefficient_variation = (
            std_amount
            / max(
                mean_amount,
                EPS,
            )
        )

        similarity = math.exp(
            -coefficient_variation
        )

    # --------------------------------------------------------
    # Pass-through
    # --------------------------------------------------------

    inbound = (
        tx.groupby(
            "nameDest"
        )["amount"]
        .sum()
    )

    outbound = (
        tx.groupby(
            "nameOrig"
        )["amount"]
        .sum()
    )

    common_accounts = (
        set(inbound.index)
        .intersection(
            outbound.index
        )
    )

    ratios = []

    for account in common_accounts:

        incoming = float(
            inbound.get(
                account,
                0.0,
            )
        )

        outgoing = float(
            outbound.get(
                account,
                0.0,
            )
        )

        if incoming <= 0:
            continue

        ratio = (
            min(
                incoming,
                outgoing,
            )
            / max(
                incoming,
                outgoing,
            )
        )

        ratios.append(ratio)

    if ratios:

        pass_through = float(
            np.mean(ratios)
        )

    else:

        pass_through = 0.0

    amount_score = (
        0.50 * similarity
        + 0.50 * pass_through
    )

    return (
        _clip01(similarity),
        _clip01(pass_through),
        _clip01(amount_score),
    )


# ============================================================
# REPEATED COUNTERPARTY
# ============================================================

def repeated_counterparty_signal(
    tx: pd.DataFrame,
) -> float:

    if tx.empty:
        return 0.0

    pair_counts = (
        tx.groupby(
            [
                "nameOrig",
                "nameDest",
            ]
        )
        .size()
    )

    total_pairs = len(
        pair_counts
    )

    if total_pairs == 0:
        return 0.0

    repeated_pairs = int(
        (
            pair_counts >= 2
        ).sum()
    )

    return _clip01(
        repeated_pairs
        / total_pairs
    )


# ============================================================
# RISK SEQUENCE
# ============================================================

def risk_sequence_signal(
    tx: pd.DataFrame,
    tx_risk: dict[int, float],
) -> float:

    if tx.empty:
        return 0.0

    work = tx.copy()

    work["tx_risk"] = [
        _safe_float(
            tx_risk.get(
                int(transaction_id),
                0.0,
            )
        )
        for transaction_id
        in work["transaction_id"]
    ]

    work = (
        work
        .sort_values(
            [
                "timestamp",
                "transaction_id",
            ]
        )
    )

    if len(work) == 1:

        return float(
            work["tx_risk"].iloc[0]
            >= 0.70
        )

    risks = (
        work["tx_risk"]
        .to_numpy(
            dtype=float
        )
    )

    high = (
        risks >= 0.70
    )

    high_count = int(
        high.sum()
    )

    if high_count == 0:
        return 0.0

    # Longest contiguous run.
    padded = np.concatenate(
        [
            [False],
            high,
            [False],
        ]
    )

    starts = np.flatnonzero(
        ~padded[:-1]
        & padded[1:]
    )

    ends = np.flatnonzero(
        padded[:-1]
        & ~padded[1:]
    )

    run_lengths = (
        ends - starts
    )

    longest_run = (
        int(
            run_lengths.max()
        )
        if len(run_lengths)
        else 0
    )

    high_fraction = (
        high_count
        / len(risks)
    )

    run_score = (
        longest_run
        / len(risks)
    )

    return _clip01(
        0.60 * high_fraction
        + 0.40 * run_score
    )


# ============================================================
# ONE CANDIDATE ANALYSIS
# ============================================================

def analyze_candidate(
    candidate_id: int,
    row: pd.Series,
    graph: nx.MultiDiGraph,
    transaction_lookup: dict[int, dict[str, Any]],
    node_risk: dict[str, float],
    tx_risk: dict[int, float],
    accounts_column: str,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Candidate accounts
    # --------------------------------------------------------

    candidate_accounts = _parse_accounts(
        row[accounts_column]
    )

    # Only keep accounts that are actually present.
    candidate_accounts = [
        account
        for account in candidate_accounts
        if account in graph
    ]

    # IMPORTANT:
    # We do not merge this candidate with another candidate.
    # We do not expand its neighborhood.
    # We do not generate another graph.
    # We analyze exactly this topology result.

    pattern_type = (
        _canonical_pattern_type(
            row
        )
    )

    # --------------------------------------------------------
    # Original transactions
    # --------------------------------------------------------

    tx = recover_candidate_transactions(
        graph=graph,
        candidate_accounts=candidate_accounts,
        transaction_lookup=transaction_lookup,
    )

    num_accounts = len(
        candidate_accounts
    )

    num_transactions = len(
        tx
    )

    # --------------------------------------------------------
    # Node risk
    # --------------------------------------------------------

    (
        mean_node_risk,
        max_node_risk,
        high_risk_node_fraction,
        node_risk_combined,
    ) = node_risk_components(
        candidate_accounts,
        node_risk,
    )

    # --------------------------------------------------------
    # Transaction risk
    # --------------------------------------------------------

    (
        mean_transaction_risk,
        max_transaction_risk,
        high_risk_transaction_fraction,
        transaction_risk_combined,
    ) = transaction_risk_components(
        tx["transaction_id"].tolist(),
        tx_risk,
    )

    # --------------------------------------------------------
    # Topology
    # --------------------------------------------------------

    topology_score = topology_strength(
        pattern_type,
        num_accounts,
    )

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    (
        rapid_score,
        layering_score,
        temporal_score,
        time_span_days,
    ) = temporal_signal(
        tx
    )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    (
        amount_similarity,
        pass_through_score,
        amount_score,
    ) = amount_signal(
        tx
    )

    # --------------------------------------------------------
    # Structural
    # --------------------------------------------------------

    structural_score = structural_signal(
        tx,
        candidate_accounts,
        pattern_type,
    )

    # --------------------------------------------------------
    # Additional diagnostics
    # --------------------------------------------------------

    repeated_counterparty_score = (
        repeated_counterparty_signal(
            tx
        )
    )

    risk_sequence_score = (
        risk_sequence_signal(
            tx,
            tx_risk,
        )
    )

    # --------------------------------------------------------
    # FINAL RING SCORE
    # --------------------------------------------------------

    ring_score = (

        0.25 * node_risk_combined

        + 0.25 * transaction_risk_combined

        + 0.15 * topology_score

        + 0.15 * temporal_score

        + 0.10 * amount_score

        + 0.10 * structural_score
    )

    ring_score = _clip01(
        ring_score
    )

    predicted_ring_alert = bool(
        ring_score >= ALERT_THRESHOLD
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------
    #
    # Ground truth is NOT read from graph edges.
    #
    # Graph edge:
    #     transaction_id
    #
    # Transaction table:
    #     transaction_id -> isFraud
    #
    # isFraud == isMoneyLaundering
    # for every transaction in the current dataset.
    # --------------------------------------------------------

    ground_truth_transactions = (
        int(
            tx["isFraud"].sum()
        )
        if not tx.empty
        else 0
    )

    contains_ground_truth = bool(
        ground_truth_transactions > 0
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if (
        predicted_ring_alert
        and contains_ground_truth
    ):

        alert_class = "TP"

    elif (
        predicted_ring_alert
        and not contains_ground_truth
    ):

        alert_class = "FP"

    elif (
        not predicted_ring_alert
        and contains_ground_truth
    ):

        alert_class = "FN"

    else:

        alert_class = "TN"

    # --------------------------------------------------------
    # Top risky nodes
    # --------------------------------------------------------

    risky_nodes = [

        {
            "account_id": account,
            "risk_score": round(
                _safe_float(
                    node_risk.get(
                        account,
                        0.0,
                    )
                ),
                4,
            ),
        }

        for account
        in candidate_accounts

        if _safe_float(
            node_risk.get(
                account,
                0.0,
            )
        ) >= 0.70
    ]

    risky_nodes = sorted(
        risky_nodes,
        key=lambda item:
            item["risk_score"],
        reverse=True,
    )[:10]

    # --------------------------------------------------------
    # Top risky transactions
    # --------------------------------------------------------

    risky_transactions = []

    for transaction_id in tx[
        "transaction_id"
    ].tolist():

        score = _safe_float(
            tx_risk.get(
                int(transaction_id),
                0.0,
            )
        )

        risky_transactions.append(
            {
                "transaction_id": int(
                    transaction_id
                ),
                "risk_score": round(
                    score,
                    6,
                ),
            }
        )

    risky_transactions = sorted(
        risky_transactions,
        key=lambda item:
            item["risk_score"],
        reverse=True,
    )[:10]

    # --------------------------------------------------------
    # Candidate transaction IDs
    # --------------------------------------------------------

    transaction_ids = [
        int(transaction_id)
        for transaction_id
        in tx[
            "transaction_id"
        ].tolist()
    ]

    return {

        "candidate_id":
            candidate_id,

        "pattern_type":
            pattern_type,

        "num_accounts":
            num_accounts,

        "num_transactions":
            num_transactions,

        # Node risk
        "mean_node_risk":
            round(
                mean_node_risk,
                6,
            ),

        "max_node_risk":
            round(
                max_node_risk,
                6,
            ),

        "high_risk_node_fraction":
            round(
                high_risk_node_fraction,
                6,
            ),

        "node_risk_combined":
            round(
                node_risk_combined,
                6,
            ),

        # Transaction risk
        "mean_transaction_risk":
            round(
                mean_transaction_risk,
                6,
            ),

        "max_transaction_risk":
            round(
                max_transaction_risk,
                6,
            ),

        "high_risk_transaction_fraction":
            round(
                high_risk_transaction_fraction,
                6,
            ),

        "transaction_risk_combined":
            round(
                transaction_risk_combined,
                6,
            ),

        # Topology
        "topology_strength":
            round(
                topology_score,
                6,
            ),

        # Temporal
        "rapid_score":
            round(
                rapid_score,
                6,
            ),

        "layering_score":
            round(
                layering_score,
                6,
            ),

        "temporal_score":
            round(
                temporal_score,
                6,
            ),

        "time_span_days":
            round(
                time_span_days,
                6,
            )
            if pd.notna(
                time_span_days
            )
            else np.nan,

        # Amount
        "amount_similarity":
            round(
                amount_similarity,
                6,
            ),

        "pass_through_score":
            round(
                pass_through_score,
                6,
            ),

        "amount_score":
            round(
                amount_score,
                6,
            ),

        # Structure
        "structural_score":
            round(
                structural_score,
                6,
            ),

        "repeated_counterparty_score":
            round(
                repeated_counterparty_score,
                6,
            ),

        "risk_sequence_score":
            round(
                risk_sequence_score,
                6,
            ),

        # Final
        "ring_score":
            round(
                ring_score,
                6,
            ),

        "predicted_ring_alert":
            predicted_ring_alert,

        # Ground truth
        "ground_truth_laundering_transactions":
            ground_truth_transactions,

        "contains_ground_truth":
            contains_ground_truth,

        "alert_class":
            alert_class,

        # Diagnostics
        "candidate_accounts":
            json.dumps(
                candidate_accounts
            ),

        "transaction_ids":
            json.dumps(
                transaction_ids
            ),

        "risky_nodes":
            json.dumps(
                risky_nodes
            ),

        "top_risky_transactions":
            json.dumps(
                risky_transactions
            ),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    result_df: pd.DataFrame,
) -> dict[str, float]:

    if result_df.empty:

        return {

            "candidates": 0,

            "predicted_alerts": 0,

            "true_positive_alerts": 0,

            "false_positive_alerts": 0,

            "false_negative_alerts": 0,

            "true_negative_candidates": 0,

            "alert_precision": 0.0,

            "alert_recall": 0.0,

            "false_alert_rate": 0.0,

            "false_positive_rate": 0.0,

            "false_positive_cost": 0.0,
        }

    tp = int(
        (
            result_df["alert_class"]
            == "TP"
        ).sum()
    )

    fp = int(
        (
            result_df["alert_class"]
            == "FP"
        ).sum()
    )

    fn = int(
        (
            result_df["alert_class"]
            == "FN"
        ).sum()
    )

    tn = int(
        (
            result_df["alert_class"]
            == "TN"
        ).sum()
    )

    predicted_alerts = (
        tp + fp
    )

    actual_positives = (
        tp + fn
    )

    actual_negatives = (
        fp + tn
    )

    precision = (
        tp / predicted_alerts
        if predicted_alerts
        else 0.0
    )

    recall = (
        tp / actual_positives
        if actual_positives
        else 0.0
    )

    # Of all alerts raised,
    # how many were false?
    false_alert_rate = (
        fp / predicted_alerts
        if predicted_alerts
        else 0.0
    )

    # Classical FPR.
    false_positive_rate = (
        fp / actual_negatives
        if actual_negatives
        else 0.0
    )

    false_positive_cost = (
        fp
        * FALSE_POSITIVE_COST
    )

    return {

        "candidates":
            int(len(result_df)),

        "predicted_alerts":
            predicted_alerts,

        "true_positive_alerts":
            tp,

        "false_positive_alerts":
            fp,

        "false_negative_alerts":
            fn,

        "true_negative_candidates":
            tn,

        "alert_precision":
            precision,

        "alert_recall":
            recall,

        "false_alert_rate":
            false_alert_rate,

        "false_positive_rate":
            false_positive_rate,

        "false_positive_cost":
            false_positive_cost,
    }


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def run_deep_risk_analysis(
    master_graph_path: str | Path = MASTER_GRAPH_PATH,
    clean_tx_path: str | Path = CLEAN_TX_PATH,
    node_risk_path: str | Path = NODE_RISK_PATH,
    tx_risk_path: str | Path = TX_RISK_PATH,
    topology_path: str | Path | None = None,
    output_path: str | Path = OUTPUT_PATH,
    alert_threshold: float = ALERT_THRESHOLD,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> pd.DataFrame:

    master_graph_path = Path(
        master_graph_path
    )

    clean_tx_path = Path(
        clean_tx_path
    )

    node_risk_path = Path(
        node_risk_path
    )

    tx_risk_path = Path(
        tx_risk_path
    )

    output_path = Path(
        output_path
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DEEP RISK ANALYSIS"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Resolve topology file
    # --------------------------------------------------------

    if topology_path is None:

        topology_path = _first_existing(
            TOPOLOGY_CANDIDATE_PATHS
        )

    else:

        topology_path = Path(
            topology_path
        )

    print(
        f"Master graph: {master_graph_path}"
    )

    print(
        f"Clean transactions: {clean_tx_path}"
    )

    print(
        f"Node risk: {node_risk_path}"
    )

    print(
        f"Transaction risk: {tx_risk_path}"
    )

    print(
        f"Topology: {topology_path}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    graph = _load_master_graph(
        master_graph_path
    )

    clean_tx = pd.read_parquet(
        clean_tx_path
    )

    node_risk_df = pd.read_parquet(
        node_risk_path
    )

    tx_risk_df = pd.read_parquet(
        tx_risk_path
    )

    topology_df = pd.read_parquet(
        topology_path
    )

    print(
        f"\nGraph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph transactions: "
        f"{graph.number_of_edges():,}"
    )

    print(
        f"Topology candidates: "
        f"{len(topology_df):,}"
    )

    # --------------------------------------------------------
    # Validate ground truth
    # --------------------------------------------------------

    if "isFraud" not in clean_tx.columns:

        raise ValueError(
            "clean_transactions.parquet "
            "must contain isFraud."
        )

    # The dataset was verified externally as:
    #
    # isFraud == isMoneyLaundering
    #
    # exactly for all transactions.
    #
    # We therefore use isFraud as the single ground-truth label.
    #
    # No ground-truth label is added to the graph.

    total_ground_truth = int(
        pd.to_numeric(
            clean_tx["isFraud"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .sum()
    )

    print(
        f"Ground-truth suspicious "
        f"transactions: "
        f"{total_ground_truth:,}"
    )

    # --------------------------------------------------------
    # Build transaction lookup
    # --------------------------------------------------------

    transaction_lookup = (
        _build_transaction_lookup(
            clean_tx
        )
    )

    # --------------------------------------------------------
    # Node risk dictionary
    # --------------------------------------------------------

    required_node_columns = {
        "account_id",
        "risk_score",
    }

    if not required_node_columns.issubset(
        node_risk_df.columns
    ):

        raise ValueError(
            "node_risk_results.parquet "
            "must contain account_id "
            "and risk_score."
        )

    node_risk_df["account_id"] = (
        node_risk_df["account_id"]
        .astype(str)
    )

    node_risk_df["risk_score"] = (
        pd.to_numeric(
            node_risk_df["risk_score"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    node_risk = dict(
        zip(
            node_risk_df["account_id"],
            node_risk_df["risk_score"],
        )
    )

    # --------------------------------------------------------
    # Transaction risk dictionary
    # --------------------------------------------------------

    required_tx_columns = {
        "transaction_id",
        "risk_score",
    }

    if not required_tx_columns.issubset(
        tx_risk_df.columns
    ):

        raise ValueError(
            "transaction_risk_all.parquet "
            "must contain transaction_id "
            "and risk_score."
        )

    tx_risk_df["transaction_id"] = (
        pd.to_numeric(
            tx_risk_df["transaction_id"],
            errors="coerce",
        )
        .astype("Int64")
    )

    tx_risk_df = tx_risk_df.dropna(
        subset=["transaction_id"]
    )

    tx_risk_df["transaction_id"] = (
        tx_risk_df["transaction_id"]
        .astype("int64")
    )

    tx_risk_df["risk_score"] = (
        pd.to_numeric(
            tx_risk_df["risk_score"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    tx_risk = dict(
        zip(
            tx_risk_df["transaction_id"],
            tx_risk_df["risk_score"],
        )
    )

    # --------------------------------------------------------
    # Topology account column
    # --------------------------------------------------------

    accounts_column = (
        _find_account_column(
            topology_df
        )
    )

    if accounts_column is None:

        raise ValueError(
            "Could not identify account "
            "column in topology results."
        )

    print(
        f"Topology account column: "
        f"{accounts_column}"
    )

    print(
        f"Alert threshold: "
        f"{alert_threshold:.2f}"
    )

    print(
        f"False-positive cost: "
        f"₹{false_positive_cost:,.0f}"
    )

    # --------------------------------------------------------
    # Analyze candidates
    # --------------------------------------------------------

    results = []

    total_candidates = (
        len(topology_df)
    )

    for index, row in (
        topology_df.iterrows()
    ):

        candidate_id_raw = row.get(
            "candidate_id",
            index,
        )

        try:

            candidate_id = int(
                candidate_id_raw
            )

        except Exception:

            candidate_id = int(
                index
            )

        # Temporarily use function-level
        # threshold / cost below through result.
        result = analyze_candidate(
            candidate_id=candidate_id,
            row=row,
            graph=graph,
            transaction_lookup=transaction_lookup,
            node_risk=node_risk,
            tx_risk=tx_risk,
            accounts_column=accounts_column,
        )

        # Make threshold configurable at runtime.
        result["predicted_ring_alert"] = (
            result["ring_score"]
            >= alert_threshold
        )

        # Recompute candidate class after
        # configurable threshold.
        contains_gt = bool(
            result[
                "contains_ground_truth"
            ]
        )

        predicted = bool(
            result[
                "predicted_ring_alert"
            ]
        )

        if predicted and contains_gt:

            result["alert_class"] = "TP"

        elif predicted and not contains_gt:

            result["alert_class"] = "FP"

        elif not predicted and contains_gt:

            result["alert_class"] = "FN"

        else:

            result["alert_class"] = "TN"

        results.append(
            result
        )

        if (
            (index + 1) % 500 == 0
            or index + 1 == total_candidates
        ):

            print(
                f"Analyzed "
                f"{index + 1:,}/"
                f"{total_candidates:,} "
                f"candidates"
            )

    # --------------------------------------------------------
    # Result dataframe
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # Save output.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        result_df
    )

    # Replace cost using runtime parameter.
    metrics[
        "false_positive_cost"
    ] = (
        metrics[
            "false_positive_alerts"
        ]
        * false_positive_cost
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DEEP RISK ANALYSIS RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"Candidates: "
        f"{metrics['candidates']:,}"
    )

    print(
        f"Predicted alerts: "
        f"{metrics['predicted_alerts']:,}"
    )

    print(
        f"True-positive alerts: "
        f"{metrics['true_positive_alerts']:,}"
    )

    print(
        f"False-positive alerts: "
        f"{metrics['false_positive_alerts']:,}"
    )

    print(
        f"False-negative alerts: "
        f"{metrics['false_negative_alerts']:,}"
    )

    print(
        f"True-negative candidates: "
        f"{metrics['true_negative_candidates']:,}"
    )

    print(
        f"\nAlert precision: "
        f"{metrics['alert_precision']:.4f}"
    )

    print(
        f"Alert recall: "
        f"{metrics['alert_recall']:.4f}"
    )

    print(
        f"False-alert rate: "
        f"{metrics['false_alert_rate']:.4f}"
    )

    print(
        f"Classical FPR: "
        f"{metrics['false_positive_rate']:.4f}"
    )

    print(
        f"False-positive cost: "
        f"₹{metrics['false_positive_cost']:,.0f}"
    )

    # --------------------------------------------------------
    # Top alerts
    # --------------------------------------------------------

    alerts = (
        result_df[
            result_df[
                "predicted_ring_alert"
            ].astype(bool)
        ]
        .sort_values(
            "ring_score",
            ascending=False,
        )
    )

    print(
        "\nTop risky candidates:"
    )

    display_columns = [
        "candidate_id",
        "pattern_type",
        "ring_score",
        "num_accounts",
        "num_transactions",
        "mean_node_risk",
        "max_node_risk",
        "mean_transaction_risk",
        "max_transaction_risk",
        "ground_truth_laundering_transactions",
        "alert_class",
    ]

    if alerts.empty:

        print(
            "No candidates crossed "
            "the alert threshold."
        )

    else:

        print(
            alerts[
                display_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    print(
        f"\nSaved: {output_path}"
    )

    return result_df


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

def run_deep_analysis(
    *args,
    **kwargs,
) -> pd.DataFrame:

    """
    Compatibility wrapper.

    Allows the pipeline to call either:

        run_deep_risk_analysis()

    or:

        run_deep_analysis()
    """

    return run_deep_risk_analysis(
        *args,
        **kwargs,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    run_deep_risk_analysis()


if __name__ == "__main__":

    main()