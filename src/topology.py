"""
Graph topology and suspicious laundering motif detection module for Abuse Ring Sentinel.

Performs two-stage topology mining:
    1. Risk Pruning: Filters transaction graph based on account-level and transaction-level risk scores.
    2. Motif Discovery: Extracts coordinated structural patterns on the collapsed relationship graph:
        - Fan-Out: Central distributor sending funds to multiple distinct recipients (layering/smurfing).
        - Fan-In: Multiple distinct senders pooling funds into a single aggregator (mule collection).
        - Reciprocal Pairs: Bidirectional transfers between account pairs (circular layering).
        - Directed Cycles: Bounded circular flows (length 3 to 6/8) returning funds to originators.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pandas as pd


def build_node_risk_lookup(node_risk_results: pd.DataFrame) -> dict[str, float]:
    """Map account_id -> risk_score."""
    if "account_id" not in node_risk_results.columns or "risk_score" not in node_risk_results.columns:
        raise ValueError("node_risk_results must contain 'account_id' and 'risk_score'.")
    return dict(zip(node_risk_results["account_id"].astype(str), node_risk_results["risk_score"].astype(float)))


def build_transaction_risk_lookup(transaction_risk_results: pd.DataFrame) -> dict[int, float]:
    """Map transaction_id -> risk_score."""
    if "transaction_id" not in transaction_risk_results.columns or "risk_score" not in transaction_risk_results.columns:
        raise ValueError("transaction_risk_results must contain 'transaction_id' and 'risk_score'.")
    return dict(zip(transaction_risk_results["transaction_id"].astype(int), transaction_risk_results["risk_score"].astype(float)))


def build_risk_pruned_graph(
    master_graph: nx.MultiDiGraph,
    node_risk_results: pd.DataFrame,
    transaction_risk_results: pd.DataFrame,
    risk_threshold: float = 0.20,
) -> nx.MultiDiGraph:
    """
    Filter the master transaction graph to retain high-risk paths.

    An edge is retained if transaction risk OR sender node risk OR receiver node risk
    meets or exceeds `risk_threshold`.
    """
    if not (0.0 <= risk_threshold <= 1.0):
        raise ValueError("risk_threshold must be between 0.0 and 1.0.")

    node_risk = build_node_risk_lookup(node_risk_results)
    tx_risk = build_transaction_risk_lookup(transaction_risk_results)

    filtered_graph = nx.MultiDiGraph()
    retained_count = 0

    for source, target, key, data in master_graph.edges(keys=True, data=True):
        tx_id = int(data.get("transaction_id", key))
        t_risk = float(tx_risk.get(tx_id, 0.0))
        s_risk = float(node_risk.get(str(source), 0.0))
        r_risk = float(node_risk.get(str(target), 0.0))

        if t_risk >= risk_threshold or s_risk >= risk_threshold or r_risk >= risk_threshold:
            edge_attrs = dict(data)
            edge_attrs.update({
                "transaction_id": tx_id,
                "transaction_risk_score": t_risk,
                "sender_node_risk": s_risk,
                "receiver_node_risk": r_risk,
            })
            filtered_graph.add_node(source, node_risk_score=s_risk)
            filtered_graph.add_node(target, node_risk_score=r_risk)
            filtered_graph.add_edge(source, target, key=tx_id, **edge_attrs)
            retained_count += 1

    orig_edges = max(master_graph.number_of_edges(), 1)
    reduction = (1.0 - (filtered_graph.number_of_edges() / orig_edges)) * 100.0

    print(f"\n[TOPOLOGY] Risk Pruning (Threshold: {risk_threshold:.2f}):")
    print(f"  • Master Graph:   {master_graph.number_of_nodes():,} nodes | {master_graph.number_of_edges():,} edges")
    print(f"  • Pruned Graph:   {filtered_graph.number_of_nodes():,} nodes | {filtered_graph.number_of_edges():,} edges")
    print(f"  • Edge Reduction: {reduction:.2f}%")

    return filtered_graph


def build_relationship_graph(filtered_graph: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Collapse multiple transactions between account pairs into single directed relationship edges.

    Stores all transaction_ids, transaction counts, and aggregated transaction risks per pair.
    """
    rel_graph = nx.DiGraph()

    for node, data in filtered_graph.nodes(data=True):
        rel_graph.add_node(node, **data)

    for source, target, key, data in filtered_graph.edges(keys=True, data=True):
        tx_id = int(data.get("transaction_id", key))
        t_risk = float(data.get("transaction_risk_score", 0.0))

        if rel_graph.has_edge(source, target):
            edge = rel_graph[source][target]
            edge["transaction_ids"].append(tx_id)
            edge["transaction_risks"].append(t_risk)
            edge["transaction_count"] += 1
        else:
            rel_graph.add_edge(
                source,
                target,
                transaction_ids=[tx_id],
                transaction_risks=[t_risk],
                transaction_count=1,
            )

    # Compute aggregate risks on relationship edges
    for _, _, edge_data in rel_graph.edges(data=True):
        risks = edge_data["transaction_risks"]
        edge_data["max_transaction_risk"] = max(risks)
        edge_data["mean_transaction_risk"] = sum(risks) / len(risks)

    return rel_graph


def find_fan_out(graph: nx.DiGraph, min_recipients: int = 3) -> List[Dict[str, Any]]:
    """Detect fan-out motifs: one central account sending to >= min_recipients accounts."""
    patterns = []
    for center in graph.nodes:
        recipients = list(graph.successors(center))
        if len(recipients) < min_recipients:
            continue

        tx_ids = []
        for rec in recipients:
            tx_ids.extend(graph[center][rec]["transaction_ids"])

        patterns.append({
            "pattern_type": "fan_out",
            "center_account": center,
            "member_accounts": recipients,
            "edge_transaction_ids": tx_ids,
            "num_members": len(recipients),
            "num_transactions": len(tx_ids),
        })
    return patterns


def find_fan_in(graph: nx.DiGraph, min_senders: int = 3) -> List[Dict[str, Any]]:
    """Detect fan-in motifs: >= min_senders accounts pooling funds into one central collector."""
    patterns = []
    for center in graph.nodes:
        senders = list(graph.predecessors(center))
        if len(senders) < min_senders:
            continue

        tx_ids = []
        for snd in senders:
            tx_ids.extend(graph[snd][center]["transaction_ids"])

        patterns.append({
            "pattern_type": "fan_in",
            "center_account": center,
            "member_accounts": senders,
            "edge_transaction_ids": tx_ids,
            "num_members": len(senders),
            "num_transactions": len(tx_ids),
        })
    return patterns


def find_reciprocal_pairs(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """Detect reciprocal pairs: bidirectional transfer loops between account pairs (A <-> B)."""
    patterns = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for u, v in graph.edges:
        if u == v or not graph.has_edge(v, u):
            continue

        pair = tuple(sorted((str(u), str(v))))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        tx_ids = list(graph[u][v]["transaction_ids"]) + list(graph[v][u]["transaction_ids"])
        patterns.append({
            "pattern_type": "reciprocal_pair",
            "center_account": None,
            "member_accounts": [pair[0], pair[1]],
            "edge_transaction_ids": tx_ids,
            "num_members": 2,
            "num_transactions": len(tx_ids),
        })
    return patterns


def find_cycles(
    graph: nx.DiGraph,
    min_length: int = 3,
    max_length: int = 6,
) -> List[Dict[str, Any]]:
    """Detect directed cycles from min_length to max_length (circular laundering chains)."""
    if min_length < 2 or max_length < min_length:
        raise ValueError("Invalid cycle bounds.")

    patterns = []
    seen_cycles: Set[Tuple[str, ...]] = set()

    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        c_len = len(cycle)
        if c_len < min_length:
            continue

        # Canonicalize cycle rotations so (A, B, C) == (B, C, A) == (C, A, B)
        c_str = [str(n) for n in cycle]
        rotations = [tuple(c_str[i:] + c_str[:i]) for i in range(c_len)]
        canonical = min(rotations)

        if canonical in seen_cycles:
            continue
        seen_cycles.add(canonical)

        tx_ids = []
        for i in range(c_len):
            u, v = cycle[i], cycle[(i + 1) % c_len]
            tx_ids.extend(graph[u][v]["transaction_ids"])

        patterns.append({
            "pattern_type": f"{c_len}_cycle",
            "center_account": None,
            "member_accounts": list(cycle),
            "edge_transaction_ids": tx_ids,
            "num_members": c_len,
            "num_transactions": len(tx_ids),
        })
    return patterns


def find_all_patterns(
    filtered_graph: nx.MultiDiGraph,
    min_fan_members: int = 3,
    max_cycle_length: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Orchestrate motif discovery across fan-out, fan-in, reciprocal pairs, and cycles."""
    print("\n[TOPOLOGY] Discovering topological laundering candidates...")
    rel_graph = build_relationship_graph(filtered_graph)
    print(f"  • Unique Account Relationships: {rel_graph.number_of_edges():,}")

    fan_out = find_fan_out(rel_graph, min_recipients=min_fan_members)
    fan_in = find_fan_in(rel_graph, min_senders=min_fan_members)
    reciprocal = find_reciprocal_pairs(rel_graph)
    cycles = find_cycles(rel_graph, min_length=3, max_length=max_cycle_length)

    print(f"  • Fan-Out Motifs:    {len(fan_out):,}")
    print(f"  • Fan-In Motifs:     {len(fan_in):,}")
    print(f"  • Reciprocal Pairs:  {len(reciprocal):,}")
    print(f"  • Directed Cycles:   {len(cycles):,}")

    return {
        "fan_out": fan_out,
        "fan_in": fan_in,
        "reciprocal_pair": reciprocal,
        "cycles": cycles,
    }


def patterns_to_dataframe(patterns: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Convert motif dictionaries into a structured DataFrame with stable pattern_id."""
    rows = []
    pattern_id = 0
    for motif_family, pattern_list in patterns.items():
        for p in pattern_list:
            # Canonicalize pattern_type
            p_type = p.get("pattern_type", motif_family)
            rows.append({
                "pattern_id": pattern_id,
                "pattern_type": p_type,
                "center_account": p.get("center_account"),
                "member_accounts": p.get("member_accounts", []),
                "edge_transaction_ids": p.get("edge_transaction_ids", []),
                "num_members": p.get("num_members", 0),
                "num_transactions": p.get("num_transactions", 0),
            })
            pattern_id += 1
    return pd.DataFrame(rows)


def validate_patterns(pattern_df: pd.DataFrame) -> None:
    """Run sanity checks verifying candidate integrity."""
    if pattern_df.empty:
        print("[TOPOLOGY] Notice: No candidates discovered.")
        return

    reciprocal = pattern_df[pattern_df["pattern_type"] == "reciprocal_pair"]
    if not reciprocal.empty:
        unique_pairs = {tuple(sorted(m, key=str)) for m in reciprocal["member_accounts"]}
        assert len(reciprocal) == len(unique_pairs), "Duplicate reciprocal pairs detected!"

    print(f"[TOPOLOGY] Candidate Validation Passed ({len(pattern_df):,} total candidates).")


def save_filtered_graph(filtered_graph: nx.MultiDiGraph, path: str | Path) -> None:
    """Serialize the risk-pruned graph to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(filtered_graph, f)
    print(f"[SAVE] Risk-pruned graph saved to: {path}")


if __name__ == "__main__":
    from pathlib import Path
    GRAPH_PATH = Path("artifacts/master_graph.pkl")
    NODE_PATH = Path("artifacts/node_risk_results.parquet")
    TX_PATH = Path("artifacts/transaction_risk_all.parquet")

    if GRAPH_PATH.exists() and NODE_PATH.exists() and TX_PATH.exists():
        print("Testing topology.py on cached artifacts...")
        with open(GRAPH_PATH, "rb") as f:
            obj = pickle.load(f)
            master_g = obj[0] if isinstance(obj, tuple) else obj

        node_df = pd.read_parquet(NODE_PATH)
        tx_df = pd.read_parquet(TX_PATH)

        pruned_g = build_risk_pruned_graph(master_g, node_df, tx_df, risk_threshold=0.20)
        p_dict = find_all_patterns(pruned_g, min_fan_members=3, max_cycle_length=6)
        cand_df = patterns_to_dataframe(p_dict)
        validate_patterns(cand_df)
        print("Topology module verification passed successfully!")
    else:
        print("Required artifacts not found for standalone testing.")
