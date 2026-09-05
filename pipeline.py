# src/pipeline.py

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import torch

from .ingest import load_transactions
from .features import (
    build_account_features,
    build_transaction_features,
    build_sender_graph_features,
)
from .graph_builder import (
    build_graph,
    integrate_features,
)
from .pyg_builder import (networkx_to_pyg,create_temporal_node_masks)
from .models.node_embeddings import train_graphsage
from .models.node_risk_score import run_node_risk_model
from .models.Transaction_risk_score import run_xgboost
from .topology import (
    build_risk_pruned_graph,
    find_all_patterns,
    patterns_to_dataframe,
    validate_patterns,
    save_filtered_graph,
)

from src.ring_risk_score import run_deep_risk_analysis

run_deep_risk_analysis()


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Change this if your CSV has a different name.
RAW_PATH = DATA_DIR / "aml_transactions.csv"


# ============================================================
# Cache paths
# ============================================================

CLEAN_PATH = ARTIFACT_DIR / "clean_transactions.parquet"

NODE_FEATURE_PATH = (
    ARTIFACT_DIR / "node_features.parquet"
)

SENDER_GRAPH_FEATURE_PATH = (
    ARTIFACT_DIR / "sender_graph_features.parquet"
)

TRANSACTION_FEATURE_PATH = (
    ARTIFACT_DIR / "transaction_features.parquet"
)

GRAPH_PATH = ARTIFACT_DIR / "master_graph.pkl"

PYG_PATH = ARTIFACT_DIR / "pyg_data.pt"

EMBEDDING_PATH = (
    ARTIFACT_DIR / "node_embeddings.pt"
)


# ============================================================
# Helpers
# ============================================================

def save_pickle(obj, path: Path) -> None:
    """Save an object using pickle."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    """Load a pickle object."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Main pipeline
# ============================================================

def prepare_pipeline(
    raw_path: str | Path = RAW_PATH,
):
    """
    Run the complete AML feature + graph preparation pipeline.

    Pipeline order:

        raw CSV
          ↓
        ingestion
          ↓
        master NetworkX graph
          ↓
        sender graph features
          ↓
        account/node features
          ↓
        transaction features
          ↓
        integrate features into SAME graph
          ↓
        PyG conversion
          ↓
        GraphSAGE
          ↓
        node embeddings
    """

    raw_path = Path(raw_path)

    # ========================================================
    # 1. Load / cache transactions
    # ========================================================

    print("[BUILD] Loading transactions...")

    if CLEAN_PATH.exists():

        df = pd.read_parquet(
            CLEAN_PATH
        )

        print(
            "[CACHE] Loaded cleaned transactions."
        )

    else:

        df = load_transactions(
            raw_path
        )

        df.to_parquet(
            CLEAN_PATH,
            index=False,
        )

        print(
            "[SAVE] Clean transactions saved."
        )

    # --------------------------------------------------------
    # Validate transaction_id
    # --------------------------------------------------------

    if "transaction_id" not in df.columns:
        raise ValueError(
            "transaction_id is missing from the cleaned "
            "transaction DataFrame. Check ingest.py."
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError(
            "transaction_id contains duplicates."
        )

    print(
        f"[INFO] Transactions: {len(df):,}"
    )

    print(
        f"[INFO] transaction_id: "
        f"{df['transaction_id'].min()} → "
        f"{df['transaction_id'].max()}"
    )


    # ========================================================
    # 2. Build / cache MASTER graph
    # ========================================================

    print(
        "[BUILD] Building master graph..."
    )

    if GRAPH_PATH.exists():

        graph, node_to_idx, idx_to_node = (
            load_pickle(GRAPH_PATH)
        )

        print(
            "[CACHE] Loaded master graph."
        )

    else:

        graph, node_to_idx, idx_to_node = (
            build_graph(df)
        )

        save_pickle(
            (
                graph,
                node_to_idx,
                idx_to_node,
            ),
            GRAPH_PATH,
        )

        print(
            "[SAVE] Master graph saved."
        )

    print(
        f"[INFO] Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"[INFO] Graph edges: "
        f"{graph.number_of_edges():,}"
    )


    # --------------------------------------------------------
    # Graph validation
    # --------------------------------------------------------

    if graph.number_of_nodes() == 0:
        raise ValueError(
            "Master graph contains no nodes."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "Master graph contains no edges."
        )

    for _, _, key, attrs in graph.edges(
        keys=True,
        data=True,
    ):
        if "transaction_id" not in attrs:
            raise ValueError(
                "A graph edge is missing transaction_id."
            )


    # ========================================================
    # 3. Build / cache sender graph features
    #
    # IMPORTANT:
    # These features MUST be calculated from the
    # already-built master graph.
    # ========================================================

    print(
        "[BUILD] Building sender graph features..."
    )

    if SENDER_GRAPH_FEATURE_PATH.exists():

        sender_graph_features = (
            pd.read_parquet(
                SENDER_GRAPH_FEATURE_PATH
            )
        )

        print(
            "[CACHE] Loaded sender graph features."
        )

    else:

        sender_graph_features = (
            build_sender_graph_features(
                graph
            )
        )

        sender_graph_features.to_parquet(
            SENDER_GRAPH_FEATURE_PATH,
            index=False,
        )

        print(
            "[SAVE] Sender graph features saved."
        )

    print(
        "[INFO] Sender graph features: "
        f"{sender_graph_features.shape}"
    )


    # ========================================================
    # 4. Build / cache ACCOUNT / NODE features
    # ========================================================

    print(
        "[BUILD] Building node features..."
    )

    if NODE_FEATURE_PATH.exists():

        node_features = pd.read_parquet(
            NODE_FEATURE_PATH
        )

        print(
            "[CACHE] Loaded node features."
        )

    else:

        node_features = (
            build_account_features(
                df
            )
        )

        node_features.to_parquet(
            NODE_FEATURE_PATH,
            index=False,
        )

        print(
            "[SAVE] Node features saved."
        )

    print(
        "[INFO] Node features: "
        f"{node_features.shape}"
    )


    # --------------------------------------------------------
    # Validate account_id
    # --------------------------------------------------------

    if "account_id" not in node_features.columns:
        raise ValueError(
            "account_id missing from node_features."
        )


    # ========================================================
    # 5. Build / cache TRANSACTION features
    #
    # Graph features are passed into this stage.
    # ========================================================

    print(
        "[BUILD] Building transaction features..."
    )

    if TRANSACTION_FEATURE_PATH.exists():

        transaction_features = (
            pd.read_parquet(
                TRANSACTION_FEATURE_PATH
            )
        )

        print(
            "[CACHE] Loaded transaction features."
        )

    else:

        transaction_features = (
            build_transaction_features(
                df=df,
                sender_graph_features=(
                    sender_graph_features
                ),
            )
        )

        transaction_features.to_parquet(
            TRANSACTION_FEATURE_PATH,
            index=False,
        )

        print(
            "[SAVE] Transaction features saved."
        )

    print(
        "[INFO] Transaction features: "
        f"{transaction_features.shape}"
    )


    # --------------------------------------------------------
    # Validate transaction features
    # --------------------------------------------------------

    required_transaction_columns = {
        "transaction_id",
        "nameOrig",
        "nameDest",
    }

    missing_transaction_columns = (
        required_transaction_columns
        - set(transaction_features.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "Transaction features are missing: "
            f"{sorted(missing_transaction_columns)}"
        )

    if transaction_features[
        "transaction_id"
    ].duplicated().any():
        raise ValueError(
            "transaction_features contains duplicate "
            "transaction_id values."
        )


    # ========================================================
    # 6. Integrate features into SAME MASTER graph
    #
    # Nothing is rebuilt here.
    #
    # At this point we already have:
    #   graph
    #   node_features
    #   transaction_features
    #
    # So now we attach the features.
    # ========================================================

    print(
        "[BUILD] Integrating features into master graph..."
    )

    graph = integrate_features(
        graph=graph,
        node_features=node_features,
        transaction_features=transaction_features,
    )

    print(
        "[INFO] Graph feature integration complete."
    )


    # ========================================================
    # 7. Convert SAME graph to PyG
    # ========================================================

    print(
        "[BUILD] Converting graph to PyG..."
    )

    if PYG_PATH.exists():

        pyg_data = torch.load(
            PYG_PATH,
            weights_only=False,
        )

        print(
            "[CACHE] Loaded PyG graph."
        )

    else:

        pyg_data = networkx_to_pyg(
            graph=graph,
            node_features=node_features,
            transaction_features=transaction_features,
        )

        pyg_data = create_temporal_node_masks(
            pyg_data=pyg_data,
           df=df,
           node_ids=pyg_data.node_ids,
        )

        torch.save(
            pyg_data,
            PYG_PATH,
        )

        print(
            "[SAVE] PyG graph saved."
        )

    print(
        f"[INFO] PyG nodes: "
        f"{pyg_data.num_nodes:,}"
    )

    print(
        f"[INFO] PyG edges: "
        f"{pyg_data.edge_index.shape[1]:,}"
    )

    print(
        f"[INFO] Node features: "
        f"{pyg_data.x.shape}"
    )

    if hasattr(
        pyg_data,
        "edge_attr",
    ):
        print(
            f"[INFO] Edge features: "
            f"{pyg_data.edge_attr.shape}"
        )


    # ========================================================
    # 8. GraphSAGE
    # ========================================================

    print(
        "\n[GNN] Training GraphSAGE..."
    )

    model, node_embeddings = train_graphsage(
        pyg_data,
        hidden_channels=128,
        embedding_dim=64,
        epochs=100,
        lr=1e-3,
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    torch.save(
        node_embeddings,
        EMBEDDING_PATH,
    )

    print(
        "[SAVE] Node embeddings saved:"
        f" {EMBEDDING_PATH}"
    )

    print(
        f"[INFO] Embedding shape: "
        f"{tuple(node_embeddings.shape)}"
    )


    print(
        "\n[NODE-RISK] Starting node-level risk model..."
    )

    (
    node_risk_model,
    node_risk_results,
    node_risk_X,
    node_risk_y,
    node_risk_feature_names,
    node_risk_scaler,
    node_risk_metrics,
) = run_node_risk_model(
    node_features=node_features,
    node_embeddings=node_embeddings,
    train_mask=pyg_data.train_mask,
    val_mask=pyg_data.val_mask,
    test_mask=pyg_data.test_mask,
    hidden_dim=128,
    dropout=0.20,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
)

    NODE_RISK_PATH = ARTIFACT_DIR / "node_risk_results.parquet"

    node_risk_results.to_parquet(
        NODE_RISK_PATH,
        index=False,
    )

    print(
        f"[SAVE] Node risk scores saved to "
        f"{NODE_RISK_PATH}"
    )

    # ========================================================
# Transaction-level XGBoost risk
# ========================================================

    print(
        "\n[TX-RISK] Running transaction risk model..."
    )

    transaction_risk_model, transaction_risk_results ,  transaction_risk_all_results  = (
        run_xgboost(
            transaction_features
        )
    )

    print(
        "\n[TX-RISK] Top risky transactions:"
    )

    print(
        transaction_risk_results
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    TRANSACTION_RISK_PATH = (
    ARTIFACT_DIR / "transaction_risk_scores.parquet"
    )

    print(
    "\n[TX-RISK] Running transaction risk model..."
     )

    (transaction_risk_model, transaction_risk_results, transaction_risk_all_results,) = run_xgboost(
    transaction_features,
    save_path=ARTIFACT_DIR / "transaction_risk_scores.parquet",
     )
    print(
    "\n[TX-RISK] Top risky transactions:"
    )

    print(
        transaction_risk_results
        .sort_values(
            "risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

            # ========================================================
        # Topology pattern discovery
        # ========================================================

    TOPOLOGY_RISK_THRESHOLD = 0.20

    filtered_graph = build_risk_pruned_graph(
    master_graph=graph,
    node_risk_results=node_risk_results,
    transaction_risk_results=transaction_risk_results,
    risk_threshold=TOPOLOGY_RISK_THRESHOLD,
    )

    save_filtered_graph(
    filtered_graph,
    ARTIFACT_DIR / "risk_pruned_graph.pkl",
    )

    patterns = find_all_patterns(
    filtered_graph,
    min_fan_members=3,
    max_cycle_length=6,
    )

    pattern_df = patterns_to_dataframe(patterns)

    validate_patterns(pattern_df)

    pattern_df.to_parquet(
    ARTIFACT_DIR / "topology_candidates.parquet",
    index=False,
    )

    ring_results = run_deep_analysis(
    master_graph=graph,
    topology_candidates=pattern_df,
    node_risk_results=node_risk_results,
    transaction_risk_results=transaction_risk_all_results,
    )

    RING_RESULTS_PATH = (
    ARTIFACT_DIR / "ring_analysis.parquet"
    )

    ring_results.to_parquet(
    RING_RESULTS_PATH,
    index=False,
    )

    print(
    f"Saved ring analysis to: "
    f"{RING_RESULTS_PATH}"
    )
    # ========================================================
    # Return everything
    # ========================================================

    return (
    df,
    graph,
    node_features,
    sender_graph_features,
    transaction_features,
    pyg_data,
    model,
    node_embeddings,
    node_risk_model,
    node_risk_results,
    transaction_risk_model,
    transaction_risk_results,
    )


# ============================================================
# Pipeline validation
# ============================================================

def check_pipeline(
    df,
    graph,
    node_features,
    sender_graph_features,
    transaction_features,
    pyg_data,
    node_embeddings=None,
):
    """Run structural consistency checks."""

    print(
        "\n[CHECK] Running pipeline checks..."
    )

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    assert len(df) > 0

    assert "transaction_id" in df.columns

    assert (
        df["transaction_id"]
        .is_unique
    )

    print(
        "[PASS] Transaction IDs are valid."
    )


    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------

    assert len(node_features) > 0

    assert (
        "account_id"
        in node_features.columns
    )

    assert (
        node_features["account_id"]
        .is_unique
    )

    print(
        "[PASS] Node features are valid."
    )


    # --------------------------------------------------------
    # Sender graph features
    # --------------------------------------------------------

    assert len(sender_graph_features) > 0

    assert (
        len(sender_graph_features)
        == graph.number_of_nodes()
    )

    print(
        "[PASS] Sender graph features match graph nodes."
    )


    # --------------------------------------------------------
    # Transaction features
    # --------------------------------------------------------

    assert len(transaction_features) > 0

    assert (
        "transaction_id"
        in transaction_features.columns
    )

    assert (
        transaction_features[
            "transaction_id"
        ].is_unique
    )

    print(
        "[PASS] Transaction features are valid."
    )


    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    assert (
        graph.number_of_nodes()
        == len(node_features)
    )

    assert (
        graph.number_of_edges()
        == len(transaction_features)
    )

    print(
        "[PASS] NetworkX graph matches feature tables."
    )


    # --------------------------------------------------------
    # PyG
    # --------------------------------------------------------

    assert pyg_data.x is not None

    assert pyg_data.edge_index is not None

    assert pyg_data.y is not None

    assert (
        pyg_data.num_nodes
        == graph.number_of_nodes()
    )

    assert (
        pyg_data.edge_index.shape[1]
        == graph.number_of_edges()
    )

    print(
        "[PASS] PyG graph matches NetworkX graph."
    )


    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    if node_embeddings is not None:

        assert (
            node_embeddings.shape[0]
            == graph.number_of_nodes()
        )

        print(
            "[PASS] Node embeddings match graph nodes."
        )


    print(
        "\n[PASS] All pipeline checks passed."
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================================"
    )

    print(
        "        ABUSE RING SENTINEL - AML PIPELINE"
    )

    print(
        "============================================================"
    )

    (
    df,
    graph,
    node_features,
    sender_graph_features,
    transaction_features,
    pyg_data,
    model,
    node_embeddings,
    node_risk_model,
    node_risk_results,
    transaction_risk_model,
    transaction_risk_results,
        ) = prepare_pipeline(
        RAW_PATH
       )

    check_pipeline(
        df=df,
        graph=graph,
        node_features=node_features,
        sender_graph_features=(
            sender_graph_features
        ),
        transaction_features=(
            transaction_features
        ),
        pyg_data=pyg_data,
        node_embeddings=node_embeddings,
    )

    print(
        "\n[DONE] Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()