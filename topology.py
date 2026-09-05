
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set

import networkx as nx
import pandas as pd


# ============================================================
# 1. Risk lookup
# ============================================================

def build_node_risk_lookup(
    node_risk_results: pd.DataFrame,
) -> Dict[Any, float]:
    """
    Build:

        account_id -> risk_score

    for all accounts.

    We keep all scores because topology pruning uses a low
    configurable threshold.
    """

    required = {"account_id", "risk_score"}

    missing = required - set(node_risk_results.columns)

    if missing:
        raise ValueError(
            f"node_risk_results missing columns: {missing}"
        )

    return dict(
        zip(
            node_risk_results["account_id"],
            node_risk_results["risk_score"].astype(float),
        )
    )


def build_transaction_risk_lookup(
    transaction_risk_results: pd.DataFrame,
) -> Dict[Any, float]:
    """
    Build:

        transaction_id -> risk_score
    """

    required = {"transaction_id", "risk_score"}

    missing = required - set(
        transaction_risk_results.columns
    )

    if missing:
        raise ValueError(
            "transaction_risk_results missing columns: "
            f"{missing}"
        )

    return dict(
        zip(
            transaction_risk_results["transaction_id"],
            transaction_risk_results["risk_score"].astype(float),
        )
    )


# ============================================================
# 2. Light risk pruning
# ============================================================

def build_risk_pruned_graph(
    master_graph: nx.MultiDiGraph,
    node_risk_results: pd.DataFrame,
    transaction_risk_results: pd.DataFrame,
    risk_threshold: float = 0.20,
) -> nx.MultiDiGraph:
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError(
            "risk_threshold must be between 0 and 1"
        )

    node_risk = build_node_risk_lookup(
        node_risk_results
    )

    transaction_risk = build_transaction_risk_lookup(
        transaction_risk_results
    )

    filtered_graph = nx.MultiDiGraph()

    retained_transactions = 0

    # --------------------------------------------------------
    # Iterate over transactions only ONCE.
    # --------------------------------------------------------

    for source, target, key, data in master_graph.edges(
        keys=True,
        data=True,
    ):

        transaction_id = data.get(
            "transaction_id",
            key,
        )

        tx_risk = float(
            transaction_risk.get(
                transaction_id,
                0.0,
            )
        )

        sender_risk = float(
            node_risk.get(
                source,
                0.0,
            )
        )

        receiver_risk = float(
            node_risk.get(
                target,
                0.0,
            )
        )

        # ----------------------------------------------------
        # Light pruning rule
        # ----------------------------------------------------

        keep = (
            tx_risk >= risk_threshold
            or sender_risk >= risk_threshold
            or receiver_risk >= risk_threshold
        )

        if not keep:
            continue

        edge_data = dict(data)

        edge_data["transaction_id"] = transaction_id
        edge_data["transaction_risk_score"] = tx_risk
        edge_data["sender_node_risk"] = sender_risk
        edge_data["receiver_node_risk"] = receiver_risk

        filtered_graph.add_node(
            source,
            node_risk_score=sender_risk,
        )

        filtered_graph.add_node(
            target,
            node_risk_score=receiver_risk,
        )

        filtered_graph.add_edge(
            source,
            target,
            key=transaction_id,
            **edge_data,
        )

        retained_transactions += 1

    print("\n" + "=" * 60)
    print("RISK PRUNING")
    print("=" * 60)

    print(
        f"Risk threshold            : {risk_threshold:.2f}"
    )

    print(
        f"Original nodes            : "
        f"{master_graph.number_of_nodes():,}"
    )

    print(
        f"Original transactions     : "
        f"{master_graph.number_of_edges():,}"
    )

    print(
        f"Filtered nodes            : "
        f"{filtered_graph.number_of_nodes():,}"
    )

    print(
        f"Filtered transactions     : "
        f"{filtered_graph.number_of_edges():,}"
    )

    transaction_reduction = (
        1.0
        - (
            filtered_graph.number_of_edges()
            / max(
                master_graph.number_of_edges(),
                1,
            )
        )
    )

    print(
        f"Transaction reduction     : "
        f"{transaction_reduction * 100:.2f}%"
    )

    return filtered_graph


# ============================================================
# 3. MultiDiGraph -> relationship graph
# ============================================================

def build_relationship_graph(
    filtered_graph: nx.MultiDiGraph,
) -> nx.DiGraph:
    """
    Collapse multiple transactions between the same accounts.

    Example:

        A -> B tx1
        A -> B tx2
        A -> B tx3

    becomes:

        A -> B

    The edge retains:

        transaction_ids
        transaction_count
        max_transaction_risk
        mean_transaction_risk
    """

    graph = nx.DiGraph()

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    for node, data in filtered_graph.nodes(
        data=True
    ):
        graph.add_node(
            node,
            **data,
        )

    # --------------------------------------------------------
    # Collapse account relationships
    # --------------------------------------------------------

    for source, target, key, data in filtered_graph.edges(
        keys=True,
        data=True,
    ):

        transaction_id = data.get(
            "transaction_id",
            key,
        )

        tx_risk = float(
            data.get(
                "transaction_risk_score",
                0.0,
            )
        )

        if graph.has_edge(source, target):

            edge = graph[source][target]

            edge["transaction_ids"].append(
                transaction_id
            )

            edge["transaction_risks"].append(
                tx_risk
            )

            edge["transaction_count"] += 1

        else:

            graph.add_edge(
                source,
                target,
                transaction_ids=[
                    transaction_id
                ],
                transaction_risks=[
                    tx_risk
                ],
                transaction_count=1,
            )

    # --------------------------------------------------------
    # Add aggregate edge information
    # --------------------------------------------------------

    for source, target, data in graph.edges(
        data=True
    ):

        risks = data["transaction_risks"]

        data["max_transaction_risk"] = max(
            risks
        )

        data["mean_transaction_risk"] = (
            sum(risks) / len(risks)
        )

    return graph


# ============================================================
# 4. Fan-out
# ============================================================

def find_fan_out(
    graph: nx.DiGraph,
    min_recipients: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find:

        A -> B
        A -> C
        A -> D
        ...

    using UNIQUE recipients.
    """

    patterns = []

    for center in graph.nodes:

        recipients = list(
            graph.successors(center)
        )

        if len(recipients) < min_recipients:
            continue

        transaction_ids = []

        for recipient in recipients:

            transaction_ids.extend(
                graph[center][recipient][
                    "transaction_ids"
                ]
            )

        patterns.append(
            {
                "pattern_type": "fan_out",
                "center_account": center,
                "member_accounts": recipients,
                "edge_transaction_ids": transaction_ids,
                "num_members": len(recipients),
                "num_transactions": len(
                    transaction_ids
                ),
            }
        )

    return patterns


# ============================================================
# 5. Fan-in
# ============================================================

def find_fan_in(
    graph: nx.DiGraph,
    min_senders: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find:

        B -> A
        C -> A
        D -> A
        ...

    using UNIQUE senders.
    """

    patterns = []

    for center in graph.nodes:

        senders = list(
            graph.predecessors(center)
        )

        if len(senders) < min_senders:
            continue

        transaction_ids = []

        for sender in senders:

            transaction_ids.extend(
                graph[sender][center][
                    "transaction_ids"
                ]
            )

        patterns.append(
            {
                "pattern_type": "fan_in",
                "center_account": center,
                "member_accounts": senders,
                "edge_transaction_ids": transaction_ids,
                "num_members": len(senders),
                "num_transactions": len(
                    transaction_ids
                ),
            }
        )

    return patterns


# ============================================================
# 6. Reciprocal pair
# ============================================================

def find_reciprocal_pairs(
    graph: nx.DiGraph,
) -> List[Dict[str, Any]]:
    """
    Find UNIQUE:

        A -> B
        B -> A

    Multiple transactions still produce one pair.
    """

    patterns = []

    seen: Set[
        Tuple[Any, Any]
    ] = set()

    for source, target in graph.edges:

        if source == target:
            continue

        pair = tuple(
            sorted(
                (source, target),
                key=str,
            )
        )

        if pair in seen:
            continue

        if not graph.has_edge(
            target,
            source,
        ):
            continue

        seen.add(pair)

        transaction_ids = []

        transaction_ids.extend(
            graph[source][target][
                "transaction_ids"
            ]
        )

        transaction_ids.extend(
            graph[target][source][
                "transaction_ids"
            ]
        )

        patterns.append(
            {
                "pattern_type": "reciprocal_pair",
                "center_account": None,
                "member_accounts": [
                    pair[0],
                    pair[1],
                ],
                "edge_transaction_ids": transaction_ids,
                "num_members": 2,
                "num_transactions": len(
                    transaction_ids
                ),
            }
        )

    return patterns


# ============================================================
# 7. Bounded cycles 3-6
# ============================================================

def find_cycles(
    graph: nx.DiGraph,
    min_length: int = 3,
    max_length: int = 6,
) -> List[Dict[str, Any]]:
    """
    Find directed cycles from length 3 through 6.

    Because the graph has already been risk-pruned and converted
    into unique account relationships, this search is much smaller
    than running cycle enumeration on the original transaction graph.

    length_bound prevents searches beyond max_length.
    """

    if min_length < 2:
        raise ValueError(
            "min_length must be >= 2"
        )

    if max_length < min_length:
        raise ValueError(
            "max_length must be >= min_length"
        )

    patterns = []

    seen_cycles: Set[
        Tuple[str, ...]
    ] = set()

    print(
        f"\nFinding cycles "
        f"{min_length}-{max_length}..."
    )

    for cycle in nx.simple_cycles(
        graph,
        length_bound=max_length,
    ):

        cycle_length = len(cycle)

        if cycle_length < min_length:
            continue

        # ----------------------------------------------------
        # Canonicalize rotations.
        #
        # A -> B -> C -> A
        # B -> C -> A -> B
        #
        # are the same cycle.
        # ----------------------------------------------------

        cycle_strings = [
            str(node)
            for node in cycle
        ]

        rotations = []

        for i in range(cycle_length):

            rotated = tuple(
                cycle_strings[i:]
                + cycle_strings[:i]
            )

            rotations.append(rotated)

        canonical = min(rotations)

        if canonical in seen_cycles:
            continue

        seen_cycles.add(canonical)

        # ----------------------------------------------------
        # Recover transaction evidence.
        # ----------------------------------------------------

        transaction_ids = []

        for i in range(cycle_length):

            source = cycle[i]
            target = cycle[
                (i + 1) % cycle_length
            ]

            edge = graph[
                source
            ][target]

            transaction_ids.extend(
                edge[
                    "transaction_ids"
                ]
            )

        patterns.append(
            {
                "pattern_type":
                    f"{cycle_length}_cycle",

                "center_account": None,

                "member_accounts":
                    list(cycle),

                "edge_transaction_ids":
                    transaction_ids,

                "num_members":
                    cycle_length,

                "num_transactions":
                    len(transaction_ids),
            }
        )

    return patterns


# ============================================================
# 8. Run all topology detection
# ============================================================

def find_all_patterns(
    filtered_graph: nx.MultiDiGraph,
    min_fan_members: int = 3,
    max_cycle_length: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all topology detection on the ALREADY risk-pruned graph.

    No risk filtering occurs inside individual topology functions.
    This is deliberate: we filter ONCE, then analyze the resulting
    graph consistently.
    """

    print("\n" + "=" * 60)
    print("TOPOLOGY DISCOVERY")
    print("=" * 60)

    print(
        f"Filtered transaction graph nodes : "
        f"{filtered_graph.number_of_nodes():,}"
    )

    print(
        f"Filtered transaction edges        : "
        f"{filtered_graph.number_of_edges():,}"
    )

    # --------------------------------------------------------
    # Relationship graph
    # --------------------------------------------------------

    print(
        "\nBuilding account relationship graph..."
    )

    graph = build_relationship_graph(
        filtered_graph
    )

    print(
        f"Unique relationships              : "
        f"{graph.number_of_edges():,}"
    )

    results = {}

    # --------------------------------------------------------
    # Fan-out
    # --------------------------------------------------------

    print("\nFinding fan-out...")

    results["fan_out"] = find_fan_out(
        graph,
        min_recipients=min_fan_members,
    )

    print(
        f"Fan-out patterns                   : "
        f"{len(results['fan_out']):,}"
    )

    # --------------------------------------------------------
    # Fan-in
    # --------------------------------------------------------

    print("\nFinding fan-in...")

    results["fan_in"] = find_fan_in(
        graph,
        min_senders=min_fan_members,
    )

    print(
        f"Fan-in patterns                    : "
        f"{len(results['fan_in']):,}"
    )

    # --------------------------------------------------------
    # Reciprocal
    # --------------------------------------------------------

    print("\nFinding reciprocal pairs...")

    results["reciprocal_pair"] = (
        find_reciprocal_pairs(graph)
    )

    print(
        f"Reciprocal pairs                   : "
        f"{len(results['reciprocal_pair']):,}"
    )

    # --------------------------------------------------------
    # Cycles
    # --------------------------------------------------------

    results["cycles"] = find_cycles(
        graph,
        min_length=3,
        max_length=max_cycle_length,
    )

    # Count individual cycle lengths
    cycle_counts = {}

    for pattern in results["cycles"]:

        pattern_type = pattern[
            "pattern_type"
        ]

        cycle_counts[pattern_type] = (
            cycle_counts.get(
                pattern_type,
                0,
            ) + 1
        )

    for length in range(
        3,
        max_cycle_length + 1,
    ):

        key = f"{length}_cycle"

        print(
            f"{length}-cycles                       : "
            f"{cycle_counts.get(key, 0):,}"
        )

    print("\n" + "=" * 60)
    print("TOPOLOGY DISCOVERY COMPLETE")
    print("=" * 60)

    return results


# ============================================================
# 9. Convert to DataFrame
# ============================================================

def patterns_to_dataframe(
    patterns: Dict[
        str,
        List[Dict[str, Any]],
    ],
) -> pd.DataFrame:
    """
    Convert topology candidates into one DataFrame.
    """

    rows = []

    pattern_id = 0

    for pattern_type, pattern_list in patterns.items():

        for pattern in pattern_list:

            rows.append(
                {
                    "pattern_id":
                        pattern_id,

                    "pattern_type":
                        pattern_type,

                    "center_account":
                        pattern.get(
                            "center_account"
                        ),

                    "member_accounts":
                        pattern.get(
                            "member_accounts",
                            [],
                        ),

                    "edge_transaction_ids":
                        pattern.get(
                            "edge_transaction_ids",
                            [],
                        ),

                    "num_members":
                        pattern.get(
                            "num_members",
                            0,
                        ),

                    "num_transactions":
                        pattern.get(
                            "num_transactions",
                            0,
                        ),
                }
            )

            pattern_id += 1

    return pd.DataFrame(rows)


# ============================================================
# 10. Validation
# ============================================================

def validate_patterns(
    pattern_df: pd.DataFrame,
) -> None:
    """
    Basic sanity checks.
    """

    if pattern_df.empty:

        print(
            "\nNo topology patterns found."
        )

        return

    print("\n" + "=" * 60)
    print("TOPOLOGY VALIDATION")
    print("=" * 60)

    print(
        pattern_df[
            "pattern_type"
        ].value_counts().to_string()
    )

    # --------------------------------------------------------
    # Reciprocal uniqueness
    # --------------------------------------------------------

    reciprocal = pattern_df[
        pattern_df["pattern_type"]
        == "reciprocal_pair"
    ]

    if not reciprocal.empty:

        unique_pairs = set()

        for members in reciprocal[
            "member_accounts"
        ]:

            unique_pairs.add(
                tuple(
                    sorted(
                        members,
                        key=str,
                    )
                )
            )

        print(
            "\nReciprocal pairs:"
        )

        print(
            f"Patterns : {len(reciprocal):,}"
        )

        print(
            f"Unique   : {len(unique_pairs):,}"
        )

        assert (
            len(reciprocal)
            == len(unique_pairs)
        ), (
            "Duplicate reciprocal pairs detected."
        )

        print(
            "PASS: reciprocal pairs are unique."
        )

    # --------------------------------------------------------
    # Fan-in uniqueness
    # --------------------------------------------------------

    fan_in = pattern_df[
        pattern_df["pattern_type"]
        == "fan_in"
    ]

    invalid_fan_in = 0

    for members in fan_in[
        "member_accounts"
    ]:

        if len(members) != len(
            set(members)
        ):
            invalid_fan_in += 1

    print(
        f"\nInvalid fan-in patterns   : "
        f"{invalid_fan_in:,}"
    )

    # --------------------------------------------------------
    # Fan-out uniqueness
    # --------------------------------------------------------

    fan_out = pattern_df[
        pattern_df["pattern_type"]
        == "fan_out"
    ]

    invalid_fan_out = 0

    for members in fan_out[
        "member_accounts"
    ]:

        if len(members) != len(
            set(members)
        ):
            invalid_fan_out += 1

    print(
        f"Invalid fan-out patterns  : "
        f"{invalid_fan_out:,}"
    )


# ============================================================
# 11. Save graph
# ============================================================

def save_filtered_graph(
    filtered_graph: nx.MultiDiGraph,
    path,
) -> None:
    """
    Save the risk-pruned transaction graph.
    """

    import pickle

    with open(path, "wb") as f:
        pickle.dump(
            filtered_graph,
            f,
        )

    print(
        f"\nSaved filtered graph to: "
        f"{path}"
    )