"""
Data ingestion and normalization module for Abuse Ring Sentinel.

Loads raw AMLNet transaction data, parses nested metadata (including timestamps,
locations, device fingerprints, and risk indicators) in a single high-efficiency
pass, enforces strict data schemas, and assigns stable transaction IDs.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: list[str] = [
    "nameOrig",
    "nameDest",
    "amount",
    "type",
    "metadata",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
]

# Safe evaluation namespace for AMLNet's string-serialized metadata
_SAFE_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "datetime": datetime,
}


def parse_metadata_record(raw_metadata: Any) -> dict[str, Any]:
    """Safely parse a single AMLNet metadata string into a dictionary."""
    if not isinstance(raw_metadata, str) or not raw_metadata.strip():
        return {}
    try:
        return eval(raw_metadata, _SAFE_EVAL_GLOBALS)
    except Exception:
        return {}


def extract_metadata_fields(metadata_series: pd.Series) -> pd.DataFrame:
    """
    Extract nested metadata fields in a single memory-efficient pass.

    Avoids multiple sequential .apply() loops over millions of records and
    prevents storing redundant in-memory Python dictionaries.
    """
    records: Dict[str, list[Any]] = {
        "timestamp": [],
        "city": [],
        "state": [],
        "country": [],
        "postcode": [],
        "device_type": [],
        "device_os": [],
        "ip_address": [],
        "payment_method": [],
        "risk_score": [],
        "customer_risk_score": [],
        "amount_vs_average": [],
        "sender_degree_metadata": [],
        "sender_clustering_metadata": [],
    }

    for raw in metadata_series:
        meta = parse_metadata_record(raw)
        loc = meta.get("location") or {}
        dev = meta.get("device_info") or {}
        risk = meta.get("risk_indicators") or {}
        net = meta.get("network_metrics") or {}

        records["timestamp"].append(meta.get("timestamp"))
        records["city"].append(loc.get("city"))
        records["state"].append(loc.get("state"))
        records["country"].append(loc.get("country"))
        records["postcode"].append(loc.get("postcode"))
        records["device_type"].append(dev.get("type"))
        records["device_os"].append(dev.get("os"))
        records["ip_address"].append(dev.get("ip_address"))
        records["payment_method"].append(meta.get("payment_method"))
        records["risk_score"].append(risk.get("risk_score"))
        records["customer_risk_score"].append(risk.get("customer_risk_score"))
        records["amount_vs_average"].append(risk.get("amount_vs_average"))
        records["sender_degree_metadata"].append(net.get("sender_degree"))
        records["sender_clustering_metadata"].append(net.get("sender_clustering"))

    extracted_df = pd.DataFrame(records)

    # Convert numeric fields efficiently
    extracted_df["timestamp"] = pd.to_datetime(extracted_df["timestamp"], errors="coerce")
    for num_col in [
        "risk_score",
        "customer_risk_score",
        "amount_vs_average",
        "sender_degree_metadata",
        "sender_clustering_metadata",
    ]:
        extracted_df[num_col] = pd.to_numeric(extracted_df[num_col], errors="coerce")

    return extracted_df


def load_transactions(
    path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load, validate, and normalize AMLNet raw transaction data.

    Parameters
    ----------
    path : str or Path
        Path to raw AMLNet CSV.
    nrows : int, optional
        Number of rows to read (useful for testing and development).

    Returns
    -------
    pd.DataFrame
        Normalized transactions with stable integer transaction_id and
        flattened metadata attributes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transaction data file not found: {path}")

    df = pd.read_csv(path, nrows=nrows)

    # 1. Validate required raw schema
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw CSV missing required columns: {sorted(missing_cols)}")

    # 2. Basic cleanup & drop records with missing crucial identifiers
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["nameOrig", "nameDest", "amount"]).reset_index(drop=True)

    # 3. Assign canonical transaction_id
    df.insert(0, "transaction_id", np.arange(len(df), dtype=np.int64))

    # 4. Extract metadata fields in a single consolidated pass
    meta_df = extract_metadata_fields(df["metadata"])

    # 5. Attach extracted fields
    df = pd.concat([df, meta_df], axis=1)

    # Ensure timestamp has valid values and sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parent.parent
    default_candidates = [
        project_root / "data" / "raw" / "amlnet" / "AMLNet_v2_transactions.csv",
        project_root / "data" / "aml_transactions.csv",
        project_root / "data" / "AMLNet_v2_transactions.csv",
    ]
    default_input = next((str(p) for p in default_candidates if p.exists()), str(default_candidates[0]))

    parser = argparse.ArgumentParser(description="Ingest AMLNet transaction dataset.")
    parser.add_argument("--input", type=str, default=default_input, help="Path to raw CSV file.")
    parser.add_argument("--output", type=str, default=None, help="Path to save clean Parquet.")
    parser.add_argument("--nrows", type=int, default=1000, help="Number of rows to test.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.exists():
        print(f"Loading data from {args.input} (nrows={args.nrows})...")
        clean_df = load_transactions(args.input, nrows=args.nrows)
    else:
        print(f"[WARN] Raw file not found at {args.input}. Generating test synthetic sample...")
        clean_df = pd.DataFrame({
            "transaction_id": np.arange(100),
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="10min"),
            "nameOrig": [f"C{i}" for i in range(100)],
            "nameDest": [f"C{i+1}" for i in range(100)],
            "amount": np.random.uniform(10, 5000, size=100),
            "type": "TRANSFER",
            "isFraud": np.random.binomial(1, 0.05, size=100),
            "isMoneyLaundering": np.random.binomial(1, 0.05, size=100),
            "laundering_typology": "fan_out",
        })

    print(f"Successfully loaded {len(clean_df):,} transactions.")
    print("Columns:", clean_df.columns.tolist())
    print("\nSample records:")
    print(clean_df[["transaction_id", "timestamp", "nameOrig", "nameDest", "amount", "isFraud"]].head())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_parquet(out_path, index=False)
        print(f"Saved to {out_path}")
