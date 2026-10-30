"""
Deep Multi-Factor Ring Risk Scoring Engine for Abuse Ring Sentinel.

Evaluates each discovered topology candidate across six independent evidence pillars:
    1. Node Risk: Mean, maximum, and high-risk fraction of member accounts.
    2. Transaction Risk: Mean, maximum, and high-risk fraction of member transactions.
    3. Topology Strength: Structural cohesion score based on motif family (cycles, fan-in/out, reciprocal).
    4. Temporal Concentration: Activity burstiness, placement, and layering decay.
    5. Amount Behavior: Amount coefficient of variation and intermediary pass-through flow ratios.
    6. Structural Fidelity: Directed graph connectivity and edge participation ratios.

Produces composite ring risk scores, generates operational alert classifications (TP/FP/FN/TN),
and computes false positive investigation costs.
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

EPS = 1e-9
DEFAULT_ALERT_THRESHOLD = 0.40
DEFAULT_FALSE_POSITIVE_COST = 2000.0  # Operational cost in currency units per false alert


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def _clip01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _parse_account_list(val: Any) -> list[str]:
    """Parse account representations (lists, numpy arrays, or strings) into clean string lists."""
    if isinstance(val, (list, tuple, np.ndarray, set)):
        return [str(x) for x in val]
    if isinstance(val, str):
        cleaned = re.sub(r"[\[\]\'\"\,]", " ", val)
        return [part.strip() for part in cleaned.split() if part.strip()]
    return []


# =============================================================================
# Evidence Pillar 1: Node Risk Components
# =============================================================================

def node_risk_components(
    candidate_accounts: list[str],
    node_risk: dict[str, float],
) -> Tuple[float, float, float, float]:
    """Compute mean risk, max risk, high-risk fraction (>=0.70), and combined node risk."""
    if not candidate_accounts:
        return 0.0, 0.0, 0.0, 0.0

    risks = np.array([_safe_float(node_risk.get(acc, 0.0)) for acc in candidate_accounts], dtype=np.float64)
    mean_risk = float(risks.mean())
    max_risk = float(risks.max())
    high_fraction = float((risks >= 0.70).mean())
    combined = 0.50 * mean_risk + 0.30 * max_risk + 0.20 * high_fraction

    return _clip01(mean_risk), _clip01(max_risk), _clip01(high_fraction), _clip01(combined)


# =============================================================================
# Evidence Pillar 2: Transaction Risk Components
# =============================================================================

def transaction_risk_components(
    transaction_ids: Iterable[int],
    tx_risk: dict[int, float],
) -> Tuple[float, float, float, float]:
    """Compute mean risk, max risk, high-risk fraction, and combined transaction risk."""
    tx_list = list(transaction_ids)
    if not tx_list:
        return 0.0, 0.0, 0.0, 0.0

    risks = np.array([_safe_float(tx_risk.get(int(tx_id), 0.0)) for tx_id in tx_list], dtype=np.float64)
    mean_risk = float(risks.mean())
    max_risk = float(risks.max())
    high_fraction = float((risks >= 0.70).mean())
    combined = 0.40 * mean_risk + 0.40 * max_risk + 0.20 * high_fraction

    return _clip01(mean_risk), _clip01(max_risk), _clip01(high_fraction), _clip01(combined)


# =============================================================================
# Evidence Pillar 3: Topology Strength
# =============================================================================

def topology_strength(pattern_type: str, num_accounts: int) -> float:
    """Evaluate structural cohesion based on motif complexity."""
    p_lower = str(pattern_type).lower()
    cycle_weights = {
        "3_cycle": 1.00,
        "4_cycle": 0.95,
        "5_cycle": 0.90,
        "6_cycle": 0.85,
    }
    if p_lower in cycle_weights:
        return cycle_weights[p_lower]
    if "reciprocal" in p_lower:
        return 0.30
    if p_lower in {"fan_in", "fan_out"}:
        return _clip01(0.30 + 0.20 * min(num_accounts / 10.0, 1.0))
    return 0.20


# =============================================================================
# Evidence Pillar 4: Temporal Signals
# =============================================================================

def temporal_signal(tx: pd.DataFrame) -> Tuple[float, float, float, float]:
    """
    Measure temporal burstiness and exponential layering dispersion over time.

    Returns
    -------
    rapid_score : float (1.0 if span <= 1 day, else 0.0)
    layering_score : float (exponential decay past 1 day)
    temporal_score : float (weighted aggregate)
    time_span_days : float (total elapsed days between first and last transfer)
    """
    if tx.empty or tx["timestamp"].isna().all():
        return 0.0, 0.0, 0.0, 0.0

    valid_ts = tx["timestamp"].dropna().sort_values()
    if len(valid_ts) < 2:
        return 0.0, 0.0, 0.0, 0.0

    span_days = float((valid_ts.max() - valid_ts.min()).total_seconds() / 86400.0)
    rapid_score = 1.0 if span_days <= 1.0 else 0.0
    layering_score = math.exp(-max(span_days - 1.0, 0.0) / 20.0)
    placement_score = math.exp(-max(span_days - 3.0, 0.0) / 20.0)

    temporal_score = 0.45 * rapid_score + 0.30 * layering_score + 0.25 * placement_score
    return _clip01(rapid_score), _clip01(layering_score), _clip01(temporal_score), span_days


# =============================================================================
# Evidence Pillar 5: Amount Behavior Signals
# =============================================================================

def amount_signal(tx: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Evaluate amount coefficient of variation (smurfing similarity) and pass-through flow ratios.

    Returns
    -------
    amount_similarity : float
    pass_through_score : float
    amount_score : float
    """
    if tx.empty:
        return 0.0, 0.0, 0.0

    amounts = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    positive_amts = amounts[amounts > 0]

    # 1. Amount similarity via negative exponential of coefficient of variation
    if len(positive_amts) <= 1:
        similarity = 0.0
    else:
        mean_amt = float(positive_amts.mean())
        std_amt = float(positive_amts.std(ddof=0))
        cv = std_amt / max(mean_amt, EPS)
        similarity = math.exp(-cv)

    # 2. Intermediary pass-through flow conservation (inbound vs outbound ratio)
    inbound = tx.groupby("nameDest")["amount"].sum()
    outbound = tx.groupby("nameOrig")["amount"].sum()
    common_accounts = set(inbound.index).intersection(outbound.index)

    ratios = []
    for acc in common_accounts:
        in_val, out_val = float(inbound[acc]), float(outbound[acc])
        if in_val > 0 and out_val > 0:
            ratios.append(min(in_val, out_val) / max(in_val, out_val))

    pass_through = float(np.mean(ratios)) if ratios else 0.0
    amount_score = 0.50 * similarity + 0.50 * pass_through

    return _clip01(similarity), _clip01(pass_through), _clip01(amount_score)


# =============================================================================
# Evidence Pillar 6: Structural Fidelity Signals
# =============================================================================

def structural_signal(
    tx: pd.DataFrame,
    candidate_accounts: list[str],
    pattern_type: str,
) -> float:
    """Assess whether member transactions actively form the claimed structural motif."""
    if tx.empty:
        return 0.0

    p_lower = str(pattern_type).lower()
    edges = set(zip(tx["nameOrig"].astype(str), tx["nameDest"].astype(str)))

    if p_lower.endswith("_cycle"):
        nums = re.findall(r"\d+", p_lower)
        if not nums:
            return 0.0
        cycle_len = int(nums[0])
        acc_set = set(candidate_accounts)
        H = nx.DiGraph()
        H.add_nodes_from(acc_set)
        H.add_edges_from(edges)
        try:
            cyclic_nodes = set()
            for c in nx.simple_cycles(H, length_bound=min(cycle_len, 6)):
                if len(c) == cycle_len:
                    cyclic_nodes.update(c)
            return _clip01(len(cyclic_nodes) / max(len(acc_set), 1))
        except Exception:
            return 0.0

    if "reciprocal" in p_lower:
        if not edges:
            return 0.0
        unique_pairs = {tuple(sorted((u, v))) for u, v in edges}
        reciprocal_count = sum(1 for u, v in edges if (v, u) in edges and u != v) // 2
        return _clip01(reciprocal_count / max(len(unique_pairs), 1))

    if p_lower in {"fan_in", "fan_out"}:
        col = "nameDest" if p_lower == "fan_in" else "nameOrig"
        counts = tx[col].value_counts()
        if counts.empty:
            return 0.0
        return _clip01(float(counts.max()) / max(float(counts.sum()), 1.0))

    return 0.0


def repeated_counterparty_signal(tx: pd.DataFrame) -> float:
    """Fraction of unique account pairs that have transacted >= 2 times."""
    if tx.empty:
        return 0.0
    pair_counts = tx.groupby(["nameOrig", "nameDest"]).size()
    total_pairs = len(pair_counts)
    if total_pairs == 0:
        return 0.0
    return _clip01(int((pair_counts >= 2).sum()) / total_pairs)


def risk_sequence_signal(tx: pd.DataFrame, tx_risk: dict[int, float]) -> float:
    """Assess consecutive bursts of high-risk transactions sorted by timestamp."""
    if tx.empty:
        return 0.0

    work = tx.sort_values(["timestamp", "transaction_id"])
    risks = np.array([_safe_float(tx_risk.get(int(t_id), 0.0)) for t_id in work["transaction_id"]])
    high = risks >= 0.70
    high_count = int(high.sum())
    if high_count == 0:
        return 0.0

    padded = np.concatenate([[False], high, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    run_lengths = ends - starts
    longest_run = int(run_lengths.max()) if len(run_lengths) else 0

    high_fraction = high_count / len(risks)
    run_score = longest_run / len(risks)
    return _clip01(0.60 * high_fraction + 0.40 * run_score)


# =============================================================================
# Transaction Recovery for Subgraph Candidates
# =============================================================================

def recover_candidate_transactions(
    graph: nx.MultiDiGraph,
    candidate_accounts: list[str],
    transaction_lookup: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """Recover original raw transaction rows linking accounts within the candidate set."""
    acc_set = set(candidate_accounts)
    rows = []
    seen_tx_ids = set()

    for u in acc_set:
        if u not in graph:
            continue
        for v in graph.successors(u):
            if v not in acc_set:
                continue
            edge_dict = graph.get_edge_data(u, v) or {}
            for _, attrs in edge_dict.items():
                tx_id = attrs.get("transaction_id")
                if tx_id is None:
                    continue
                tx_id = int(tx_id)
                if tx_id in seen_tx_ids:
                    continue
                seen_tx_ids.add(tx_id)

                tx_data = transaction_lookup.get(tx_id)
                if tx_data:
                    row_dict = dict(tx_data)
                    row_dict["transaction_id"] = tx_id
                    rows.append(row_dict)

    if not rows:
        return pd.DataFrame(columns=["transaction_id", "nameOrig", "nameDest", "amount", "timestamp", "isFraud"])

    df_res = pd.DataFrame(rows)
    df_res["timestamp"] = pd.to_datetime(df_res["timestamp"], errors="coerce")
    df_res["amount"] = pd.to_numeric(df_res["amount"], errors="coerce").fillna(0.0)
    df_res["isFraud"] = pd.to_numeric(df_res.get("isFraud", 0), errors="coerce").fillna(0).astype(np.int8)
    return df_res.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


# =============================================================================
# Candidate Analysis Routine
# =============================================================================

def analyze_candidate(
    candidate_id: int,
    row: pd.Series,
    graph: nx.MultiDiGraph,
    transaction_lookup: dict[int, dict[str, Any]],
    node_risk: dict[str, float],
    tx_risk: dict[int, float],
    accounts_column: str = "member_accounts",
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
) -> dict[str, Any]:
    """Perform deep 6-pillar analysis on a single topological laundering candidate."""
    raw_accounts = _parse_account_list(row[accounts_column])
    candidate_accounts = [acc for acc in raw_accounts if acc in graph]
    pattern_type = str(row.get("pattern_type", "unknown"))

    tx = recover_candidate_transactions(graph, candidate_accounts, transaction_lookup)
    num_accounts = len(candidate_accounts)
    num_transactions = len(tx)

    # 1. Node Risk
    mean_nr, max_nr, high_nr_frac, node_risk_comb = node_risk_components(candidate_accounts, node_risk)

    # 2. Transaction Risk
    tx_ids = tx["transaction_id"].tolist()
    mean_tr, max_tr, high_tr_frac, tx_risk_comb = transaction_risk_components(tx_ids, tx_risk)

    # 3. Topology Strength
    topo_score = topology_strength(pattern_type, num_accounts)

    # 4. Temporal Dispersion
    rapid_s, layer_s, temp_score, time_span_days = temporal_signal(tx)

    # 5. Amount Behavior
    amt_sim, pass_thru, amt_score = amount_signal(tx)

    # 6. Structural Fidelity & Diagnostics
    struct_score = structural_signal(tx, candidate_accounts, pattern_type)
    repeated_score = repeated_counterparty_signal(tx)
    risk_seq_score = risk_sequence_signal(tx, tx_risk)

    # Composite Ring Risk Score
    ring_score = _clip01(
        0.25 * node_risk_comb
        + 0.25 * tx_risk_comb
        + 0.15 * topo_score
        + 0.15 * temp_score
        + 0.10 * amt_score
        + 0.10 * struct_score
    )

    predicted_alert = bool(ring_score >= alert_threshold)
    gt_count = int(tx["isFraud"].sum()) if not tx.empty else 0
    contains_gt = bool(gt_count > 0)

    # Confusion matrix alert class
    if predicted_alert and contains_gt:
        alert_class = "TP"
    elif predicted_alert and not contains_gt:
        alert_class = "FP"
    elif not predicted_alert and contains_gt:
        alert_class = "FN"
    else:
        alert_class = "TN"

    # Risky nodes & transactions profiling
    risky_nodes = sorted(
        [
            {"account_id": acc, "risk_score": round(_safe_float(node_risk.get(acc, 0.0)), 4)}
            for acc in candidate_accounts if _safe_float(node_risk.get(acc, 0.0)) >= 0.70
        ],
        key=lambda x: x["risk_score"],
        reverse=True,
    )[:10]

    top_risky_tx = sorted(
        [
            {"transaction_id": int(t_id), "risk_score": round(_safe_float(tx_risk.get(int(t_id), 0.0)), 6)}
            for t_id in tx_ids
        ],
        key=lambda x: x["risk_score"],
        reverse=True,
    )[:10]

    return {
        "candidate_id": candidate_id,
        "pattern_type": pattern_type,
        "num_accounts": num_accounts,
        "num_transactions": num_transactions,
        "mean_node_risk": mean_nr,
        "max_node_risk": max_nr,
        "high_risk_node_fraction": high_nr_frac,
        "node_risk_combined": node_risk_comb,
        "mean_transaction_risk": mean_tr,
        "max_transaction_risk": max_tr,
        "high_risk_transaction_fraction": high_tr_frac,
        "transaction_risk_combined": tx_risk_comb,
        "topology_strength": topo_score,
        "rapid_score": rapid_s,
        "layering_score": layer_s,
        "temporal_score": temp_score,
        "time_span_days": time_span_days,
        "amount_similarity": amt_sim,
        "pass_through_score": pass_thru,
        "amount_score": amt_score,
        "structural_score": struct_score,
        "repeated_counterparty_score": repeated_score,
        "risk_sequence_score": risk_seq_score,
        "ring_score": ring_score,
        "predicted_ring_alert": predicted_alert,
        "ground_truth_laundering_transactions": gt_count,
        "contains_ground_truth": contains_gt,
        "alert_class": alert_class,
        "candidate_accounts": candidate_accounts,
        "transaction_ids": tx_ids,
        "risky_nodes": risky_nodes,
        "top_risky_transactions": top_risky_tx,
    }


# =============================================================================
# High-Level Deep Risk Analysis Orchestrator
# =============================================================================

def calculate_ring_metrics(result_df: pd.DataFrame, false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST) -> dict[str, float]:
    """Calculate operational metrics, alert precision, recall, and total false-alert cost."""
    if result_df.empty:
        return {"candidates": 0, "predicted_alerts": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 0.0, "recall": 0.0, "fp_cost": 0.0}

    tp = int((result_df["alert_class"] == "TP").sum())
    fp = int((result_df["alert_class"] == "FP").sum())
    fn = int((result_df["alert_class"] == "FN").sum())
    tn = int((result_df["alert_class"] == "TN").sum())

    predicted = tp + fp
    actual_pos = tp + fn
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual_pos if actual_pos else 0.0
    fp_cost = fp * false_positive_cost

    return {
        "candidates": len(result_df),
        "predicted_alerts": predicted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_cost": fp_cost,
    }


def run_deep_risk_analysis(
    master_graph: nx.MultiDiGraph | str | Path,
    clean_transactions: pd.DataFrame | str | Path,
    node_risk_results: pd.DataFrame | str | Path,
    transaction_risk_results: pd.DataFrame | str | Path,
    topology_candidates: pd.DataFrame | str | Path,
    output_path: str | Path | None = None,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    """
    Execute deep multi-factor risk scoring on all discovered topology candidates.

    Accepts either in-memory objects or file paths for maximum flexibility.
    """
    # 1. Resolve master graph
    if isinstance(master_graph, (str, Path)):
        with open(master_graph, "rb") as f:
            obj = pickle.load(f)
            graph = obj[0] if isinstance(obj, tuple) else obj
    else:
        graph = master_graph

    # 2. Resolve DataFrames
    clean_tx = pd.read_parquet(clean_transactions) if isinstance(clean_transactions, (str, Path)) else clean_transactions
    node_risk_df = pd.read_parquet(node_risk_results) if isinstance(node_risk_results, (str, Path)) else node_risk_results
    tx_risk_df = pd.read_parquet(transaction_risk_results) if isinstance(transaction_risk_results, (str, Path)) else transaction_risk_results
    topo_df = pd.read_parquet(topology_candidates) if isinstance(topology_candidates, (str, Path)) else topology_candidates

    # 3. Build lookups
    tx_risk = dict(zip(tx_risk_df["transaction_id"].astype(int), tx_risk_df["risk_score"].astype(float)))
    node_risk = dict(zip(node_risk_df["account_id"].astype(str), node_risk_df["risk_score"].astype(float)))

    # Fast indexed lookup for transactions
    clean_indexed = clean_tx.set_index("transaction_id")[["nameOrig", "nameDest", "amount", "timestamp", "isFraud"]]
    tx_lookup = clean_indexed.to_dict(orient="index")

    # Determine accounts column name
    acc_col = "member_accounts" if "member_accounts" in topo_df.columns else "candidate_accounts"

    print(f"\n[RING-SCORING] Analyzing {len(topo_df):,} candidates (Alert Threshold: {alert_threshold:.2f})...")
    results = []

    for idx, row in topo_df.iterrows():
        cand_id = int(row.get("pattern_id", row.get("candidate_id", idx)))
        res = analyze_candidate(
            candidate_id=cand_id,
            row=row,
            graph=graph,
            transaction_lookup=tx_lookup,
            node_risk=node_risk,
            tx_risk=tx_risk,
            accounts_column=acc_col,
            alert_threshold=alert_threshold,
        )
        results.append(res)

        if (idx + 1) % 2500 == 0 or (idx + 1) == len(topo_df):
            print(f"  • Processed {idx + 1:,} / {len(topo_df):,} candidates...")

    result_df = pd.DataFrame(results)

    # 4. Compute and display operational metrics
    metrics = calculate_ring_metrics(result_df, false_positive_cost)
    print("\n" + "=" * 65)
    print("RING SENTINEL OPERATIONAL EVALUATION")
    print("=" * 65)
    print(f"  Total Candidates Analyzed: {metrics['candidates']:,}")
    print(f"  Predicted Ring Alerts:     {metrics['predicted_alerts']:,}")
    print(f"  True Positives (TP):       {metrics['true_positives']:,}")
    print(f"  False Positives (FP):      {metrics['false_positives']:,}")
    print(f"  False Negatives (FN):      {metrics['false_negatives']:,}")
    print(f"  Alert Precision:           {metrics['precision']:.4f}")
    print(f"  Alert Recall:              {metrics['recall']:.4f}")
    print(f"  Total Operational FP Cost: ₹{metrics['false_positive_cost']:,.0f}")
    print("=" * 65)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(out_p, index=False)
        print(f"[SAVE] Ring analysis saved to: {out_p}")

    return result_df


# Compatibility alias
def run_deep_analysis(*args, **kwargs) -> pd.DataFrame:
    """Compatibility wrapper matching legacy import signatures."""
    # Handle parameter name variations
    if "topology_candidates" in kwargs:
        pass
    elif "topology_path" in kwargs:
        kwargs["topology_candidates"] = kwargs.pop("topology_path")
    return run_deep_risk_analysis(*args, **kwargs)


if __name__ == "__main__":
    G_PATH = Path("artifacts/master_graph.pkl")
    CLEAN_PATH = Path("artifacts/clean_transactions.parquet")
    N_PATH = Path("artifacts/node_risk_results.parquet")
    TX_PATH = Path("artifacts/transaction_risk_all.parquet")
    TOP_PATH = Path("artifacts/topology_candidates.parquet")

    if all(p.exists() for p in [G_PATH, CLEAN_PATH, N_PATH, TX_PATH, TOP_PATH]):
        print("Testing ring_risk_score.py on cached candidates sample...")
        topo_sub = pd.read_parquet(TOP_PATH).head(500)
        ring_df = run_deep_risk_analysis(
            master_graph=G_PATH,
            clean_transactions=CLEAN_PATH,
            node_risk_results=N_PATH,
            transaction_risk_results=TX_PATH,
            topology_candidates=topo_sub,
        )
        print(f"Sample ring score shape: {ring_df.shape}")
        print("Ring risk module verification passed successfully!")
    else:
        print("Required artifacts not found for testing.")
