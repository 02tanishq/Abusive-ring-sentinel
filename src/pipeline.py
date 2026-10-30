"""
Master end-to-end orchestration pipeline for Abuse Ring Sentinel.

Executes the complete multi-level AML detection architecture:
    1. Data Ingestion & Metadata Normalization
    2. Master NetworkX MultiDiGraph Construction
    3. Structural Sender Graph Features (PageRank, HITS, Degree Asymmetry)
    4. Account / Node Aggregations & Tabular Transaction Feature Engineering
    5. PyTorch Geometric Conversion & Inductive GraphSAGE GNN Training
    6. Multilayer Perceptron (MLP) Account-Level Risk Prediction
    7. Tabular XGBoost Transaction-Level Fraud Prediction
    8. Risk Pruning & Topological Motif Discovery (Cycles, Fan-In/Out, Reciprocal Pairs)
    9. Deep 6-Pillar Ring Risk Scoring & Operational Cost Analysis
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

# Ensure src directory is in Python path for clean module or script execution
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingest import load_transactions
from features import (
    build_account_features,
    build_sender_graph_features,
    build_transaction_features,
)
from graph_builder import build_graph, integrate_features
from pyg_builder import create_temporal_node_masks, networkx_to_pyg
from models.node_embeddings import train_graphsage
from models.node_risk_score import run_node_risk_model
from models.Transaction_risk_score import run_xgboost
from topology import (
    build_risk_pruned_graph,
    find_all_patterns,
    patterns_to_dataframe,
    save_filtered_graph,
    validate_patterns,
)
from ring_risk_score import run_deep_risk_analysis

# =============================================================================
# Path & Artifact Configuration
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RAW_CANDIDATES = [
    DATA_DIR / "raw" / "amlnet" / "AMLNet_v2_transactions.csv",
    DATA_DIR / "aml_transactions.csv",
    DATA_DIR / "AMLNet_v2_transactions.csv",
]

CLEAN_PATH = ARTIFACT_DIR / "clean_transactions.parquet"
GRAPH_PATH = ARTIFACT_DIR / "master_graph.pkl"
SENDER_GRAPH_PATH = ARTIFACT_DIR / "sender_graph_features.parquet"
NODE_FEATURE_PATH = ARTIFACT_DIR / "node_features.parquet"
TX_FEATURE_PATH = ARTIFACT_DIR / "transaction_features.parquet"
PYG_PATH = ARTIFACT_DIR / "pyg_data.pt"
EMBEDDING_PATH = ARTIFACT_DIR / "node_embeddings.pt"
NODE_RISK_PATH = ARTIFACT_DIR / "node_risk_results.parquet"
TX_RISK_PATH = ARTIFACT_DIR / "transaction_risk_scores.parquet"
TX_RISK_ALL_PATH = ARTIFACT_DIR / "transaction_risk_all.parquet"
PRUNED_GRAPH_PATH = ARTIFACT_DIR / "risk_pruned_graph.pkl"
TOPOLOGY_PATH = ARTIFACT_DIR / "topology_candidates.parquet"
RING_ANALYSIS_PATH = ARTIFACT_DIR / "ring_analysis.parquet"


@dataclass
class PipelineArtifacts:
    """Container holding intermediate memory representations of the pipeline."""

    transactions: pd.DataFrame
    graph: Any
    sender_features: pd.DataFrame
    node_features: pd.DataFrame
    transaction_features: pd.DataFrame
    pyg_data: Any
    node_embeddings: torch.Tensor
    node_risk_results: pd.DataFrame
    transaction_risk_results: pd.DataFrame
    transaction_risk_all: pd.DataFrame
    topology_candidates: pd.DataFrame
    ring_analysis: pd.DataFrame


def resolve_raw_path(override_path: str | Path | None = None) -> Path:
    """Find valid raw dataset location."""
    if override_path:
        p = Path(override_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified raw dataset path does not exist: {p}")

    for candidate in DEFAULT_RAW_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Raw AMLNet dataset not found. Checked: {[str(c) for c in DEFAULT_RAW_CANDIDATES]}. "
        "Specify with --raw-path."
    )


# =============================================================================
# Pipeline Execution Stages
# =============================================================================

def run_pipeline(
    raw_path: str | Path | None = None,
    force_recompute: bool = False,
    topology_risk_threshold: float = 0.20,
    ring_alert_threshold: float = 0.40,
    max_cycle_length: int = 6,
) -> PipelineArtifacts:
    """
    Run end-to-end Abuse Ring Sentinel detection pipeline with caching.

    Parameters
    ----------
    raw_path : Path, optional
    force_recompute : bool, default False
    topology_risk_threshold : float, default 0.20
    ring_alert_threshold : float, default 0.40
    max_cycle_length : int, default 6

    Returns
    -------
    PipelineArtifacts
    """
    t_start = time.time()
    print("=" * 70)
    print("       ABUSE RING SENTINEL - AML GRAPH DETECTION PIPELINE")
    print("=" * 70)

    # 1. Ingestion
    raw_csv = resolve_raw_path(raw_path)
    print(f"\n[STAGE 1/8] Ingesting transactions from: {raw_csv}")
    if CLEAN_PATH.exists() and not force_recompute:
        print("  • Loading cached clean transactions...")
        df = pd.read_parquet(CLEAN_PATH)
    else:
        df = load_transactions(raw_csv)
        df.to_parquet(CLEAN_PATH, index=False)
        print(f"  • Cached {len(df):,} normalized transactions to {CLEAN_PATH}")

    # 2. Master MultiDiGraph Construction
    print("\n[STAGE 2/8] Building canonical NetworkX master graph...")
    if GRAPH_PATH.exists() and not force_recompute:
        print("  • Loading cached master graph...")
        with open(GRAPH_PATH, "rb") as f:
            obj = pickle.load(f)
            graph = obj[0] if isinstance(obj, tuple) else obj
    else:
        graph, n2i, i2n = build_graph(df)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump((graph, n2i, i2n), f)
        print(f"  • Saved master graph ({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges)")

    # 3. Sender Graph Features
    print("\n[STAGE 3/8] Computing sender graph metrics (PageRank, HITS, flow asymmetry)...")
    if SENDER_GRAPH_PATH.exists() and not force_recompute:
        sender_features = pd.read_parquet(SENDER_GRAPH_PATH)
    else:
        sender_features = build_sender_graph_features(graph)
        sender_features.to_parquet(SENDER_GRAPH_PATH, index=False)

    # 4. Account & Transaction Tabular Features
    print("\n[STAGE 4/8] Building account and tabular transaction features...")
    if NODE_FEATURE_PATH.exists() and not force_recompute:
        node_features = pd.read_parquet(NODE_FEATURE_PATH)
    else:
        node_features = build_account_features(df)
        node_features.to_parquet(NODE_FEATURE_PATH, index=False)

    if TX_FEATURE_PATH.exists() and not force_recompute:
        tx_features = pd.read_parquet(TX_FEATURE_PATH)
    else:
        tx_features = build_transaction_features(df, sender_features)
        tx_features.to_parquet(TX_FEATURE_PATH, index=False)

    # Attach features into master graph
    graph = integrate_features(graph, node_features, tx_features)

    # 5. PyG Conversion & GraphSAGE GNN Embeddings
    print("\n[STAGE 5/8] Vectorizing PyG tensors & training GraphSAGE GNN...")
    if PYG_PATH.exists() and not force_recompute:
        pyg_data = torch.load(PYG_PATH, weights_only=False)
    else:
        pyg_data = networkx_to_pyg(graph, node_features, tx_features)
        pyg_data = create_temporal_node_masks(pyg_data, df, pyg_data.node_ids)
        torch.save(pyg_data, PYG_PATH)

    if EMBEDDING_PATH.exists() and not force_recompute:
        node_embeddings = torch.load(EMBEDDING_PATH, weights_only=False)
    else:
        _, node_embeddings = train_graphsage(pyg_data, hidden_channels=128, embedding_dim=64, epochs=100)
        torch.save(node_embeddings, EMBEDDING_PATH)

    # 6. Account-Level Node Risk MLP
    print("\n[STAGE 6/8] Training account-level MLP risk classifier...")
    if NODE_RISK_PATH.exists() and not force_recompute:
        node_risk_results = pd.read_parquet(NODE_RISK_PATH)
    else:
        _, node_risk_results, _, _, _, _, _ = run_node_risk_model(
            node_features=node_features,
            node_embeddings=node_embeddings,
            train_mask=pyg_data.train_mask,
            val_mask=pyg_data.val_mask,
            test_mask=pyg_data.test_mask,
            epochs=100,
        )
        node_risk_results.to_parquet(NODE_RISK_PATH, index=False)

    # 7. Transaction-Level XGBoost Risk Model
    print("\n[STAGE 7/8] Training transaction-level XGBoost risk classifier...")
    if TX_RISK_PATH.exists() and TX_RISK_ALL_PATH.exists() and not force_recompute:
        tx_risk_results = pd.read_parquet(TX_RISK_PATH)
        tx_risk_all = pd.read_parquet(TX_RISK_ALL_PATH)
    else:
        _, tx_risk_results, tx_risk_all = run_xgboost(
            transaction_features=tx_features,
            save_path=TX_RISK_PATH,
        )

    # 8. Graph Topology Motif Mining & Deep Ring Risk Analysis
    print("\n[STAGE 8/8] Performing topology discovery & deep ring scoring...")
    if TOPOLOGY_PATH.exists() and not force_recompute:
        topology_candidates = pd.read_parquet(TOPOLOGY_PATH)
    else:
        pruned_graph = build_risk_pruned_graph(
            master_graph=graph,
            node_risk_results=node_risk_results,
            transaction_risk_results=tx_risk_results,
            risk_threshold=topology_risk_threshold,
        )
        save_filtered_graph(pruned_graph, PRUNED_GRAPH_PATH)

        pattern_dict = find_all_patterns(
            pruned_graph,
            min_fan_members=3,
            max_cycle_length=max_cycle_length,
        )
        topology_candidates = patterns_to_dataframe(pattern_dict)
        validate_patterns(topology_candidates)
        topology_candidates.to_parquet(TOPOLOGY_PATH, index=False)

    if RING_ANALYSIS_PATH.exists() and not force_recompute:
        ring_analysis = pd.read_parquet(RING_ANALYSIS_PATH)
        if len(ring_analysis) != len(topology_candidates):
            print(f"  • Cached ring analysis count ({len(ring_analysis):,}) diverges from candidates ({len(topology_candidates):,}). Synchronizing...")
            ring_analysis = run_deep_risk_analysis(
                master_graph=graph,
                clean_transactions=df,
                node_risk_results=node_risk_results,
                transaction_risk_results=tx_risk_all,
                topology_candidates=topology_candidates,
                output_path=RING_ANALYSIS_PATH,
                alert_threshold=ring_alert_threshold,
            )
    else:
        ring_analysis = run_deep_risk_analysis(
            master_graph=graph,
            clean_transactions=df,
            node_risk_results=node_risk_results,
            transaction_risk_results=tx_risk_all,
            topology_candidates=topology_candidates,
            output_path=RING_ANALYSIS_PATH,
            alert_threshold=ring_alert_threshold,
        )

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE: Finished all stages in {elapsed_total:.2f} seconds.")
    print(f"Artifacts successfully saved to: {ARTIFACT_DIR}")
    print("=" * 70)

    return PipelineArtifacts(
        transactions=df,
        graph=graph,
        sender_features=sender_features,
        node_features=node_features,
        transaction_features=tx_features,
        pyg_data=pyg_data,
        node_embeddings=node_embeddings,
        node_risk_results=node_risk_results,
        transaction_risk_results=tx_risk_results,
        transaction_risk_all=tx_risk_all,
        topology_candidates=topology_candidates,
        ring_analysis=ring_analysis,
    )


def check_pipeline(ctx: PipelineArtifacts) -> None:
    """Execute structural consistency checks verifying complete artifact integrity."""
    print("\n[CHECK] Running pipeline validation checks...")
    assert len(ctx.transactions) > 0, "Empty transactions."
    assert ctx.graph.number_of_nodes() == len(ctx.node_features), "Node count mismatch between graph & features."
    assert ctx.pyg_data.num_nodes == ctx.graph.number_of_nodes(), "PyG node count mismatch."
    assert len(ctx.topology_candidates) > 0, "No topology candidates detected."
    assert len(ctx.ring_analysis) > 0, "Ring analysis is empty."
    print(
        f"  [PASS] All structural assertions verified successfully: "
        f"{len(ctx.transactions):,} transactions, {ctx.graph.number_of_nodes():,} accounts, "
        f"{len(ctx.topology_candidates):,} topology candidates, and {len(ctx.ring_analysis):,} ring analyses."
    )


def main():
    parser = argparse.ArgumentParser(description="Abuse Ring Sentinel Pipeline Runner")
    parser.add_argument("--raw-path", type=str, default=None, help="Custom path to raw CSV dataset.")
    parser.add_argument("--force", action="store_true", help="Force recomputation of all cached artifacts.")
    parser.add_argument("--topo-thresh", type=float, default=0.20, help="Topology risk threshold.")
    parser.add_argument("--alert-thresh", type=float, default=0.40, help="Ring alert threshold.")
    parser.add_argument("--max-cycle", type=int, default=6, help="Maximum cycle length to mine.")
    args = parser.parse_args()

    artifacts = run_pipeline(
        raw_path=args.raw_path,
        force_recompute=args.force,
        topology_risk_threshold=args.topo_thresh,
        ring_alert_threshold=args.alert_thresh,
        max_cycle_length=args.max_cycle,
    )
    check_pipeline(artifacts)


if __name__ == "__main__":
    main()
