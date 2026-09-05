import networkx as nx
import pandas as pd


def build_graph(df: pd.DataFrame):

    required_columns = [
        "transaction_id",
        "nameOrig",
        "nameDest",
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
    # Nodes
    # --------------------------------------------------------

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
        name="node_id",
    )

    node_to_idx = {
        node: idx
        for idx, node in enumerate(nodes)
    }

    idx_to_node = {
        idx: node
        for node, idx in node_to_idx.items()
    }

    # --------------------------------------------------------
    # Master graph
    # --------------------------------------------------------

    graph = nx.MultiDiGraph()

    graph.add_nodes_from(nodes)

    # Store mappings inside the graph so every downstream
    # component can access the same canonical node ordering.
    graph.graph["node_to_idx"] = node_to_idx
    graph.graph["idx_to_node"] = idx_to_node

    # --------------------------------------------------------
    # Transaction edges
    # --------------------------------------------------------

    for row in df[
        [
            "transaction_id",
            "nameOrig",
            "nameDest",
        ]
    ].itertuples(index=False):

        graph.add_edge(
            row.nameOrig,
            row.nameDest,
            key=row.transaction_id,
            transaction_id=row.transaction_id,
        )

    return graph, node_to_idx, idx_to_node


def integrate_features(
    graph: nx.MultiDiGraph,
    node_features: pd.DataFrame,
    transaction_features: pd.DataFrame,
) -> nx.MultiDiGraph:
    
    # 1. Validate node features
    

    if "account_id" not in node_features.columns:
        raise ValueError(
            "node_features must contain 'account_id'."
        )

    
    # 2. Validate transaction features
    

    if "transaction_id" not in transaction_features.columns:
        raise ValueError(
            "transaction_features must contain "
            "'transaction_id'."
        )

    
    # 3. Attach node features
    

    node_feature_columns = [
        column
        for column in node_features.columns
        if column != "account_id"
    ]

    node_lookup = (
        node_features
        .set_index("account_id")
        .to_dict(orient="index")
    )

    for node in graph.nodes:

        features = node_lookup.get(node)

        if features is None:
            continue

        for column in node_feature_columns:
            graph.nodes[node][column] = features[column]

    
    # 4. Attach transaction / edge features
    

    transaction_feature_columns = [
        column
        for column in transaction_features.columns
        if column not in {
            "transaction_id",
            "nameOrig",
            "nameDest",
        }
    ]

    transaction_lookup = (
        transaction_features
        .set_index("transaction_id")
        .to_dict(orient="index")
    )

    for source, destination, key, edge_data in graph.edges(
        keys=True,
        data=True,
    ):

        # transaction_id was stored as the edge key
        # when the master graph was created.
        transaction_id = edge_data.get(
            "transaction_id",
            key,
        )

        features = transaction_lookup.get(
            transaction_id
        )

        if features is None:
            continue

        for column in transaction_feature_columns:
            edge_data[column] = features[column]

    return graph