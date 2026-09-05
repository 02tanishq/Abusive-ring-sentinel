# src/ingest.py

import ast
import pandas as pd


REQUIRED_COLUMNS = [
    "nameOrig",
    "nameDest",
    "amount",
    "type",
    "metadata",
    "isFraud",
    "isMoneyLaundering",
    "laundering_typology",
]


def parse_metadata(value):
    """Parse AMLNet's metadata string into a dictionary."""
    if not isinstance(value, str):
        return {}

    try:
        return eval(
            value,
            {"__builtins__": {}, "datetime": __import__("datetime")},
        )
    except Exception:
        return {}


def load_transactions(path):
    """Load and normalize AMLNet transactions."""

    df = pd.read_csv(path)

    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Basic cleanup
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["nameOrig", "nameDest", "amount"])

    # Making Transaction_id
    df.insert(
        0,
        "transaction_id",
        range(len(df)),
    )

    # Parse metadata
    metadata = df["metadata"].apply(parse_metadata)

    # Extract useful metadata
    df["timestamp"] = pd.to_datetime(
        metadata.apply(lambda x: x.get("timestamp")),
        errors="coerce",
    )

    df["city"] = metadata.apply(
        lambda x: x.get("location", {}).get("city")
    )

    df["state"] = metadata.apply(
        lambda x: x.get("location", {}).get("state")
    )
    df["country"] = metadata.apply(
            lambda x: x.get("location", {}).get("country")
    )
    df["postcode"] = metadata.apply(
            lambda x: x.get("location", {}).get("postcode")
    )
    df["device_type"] = metadata.apply(
        lambda x: x.get("device_info", {}).get("type")
    )

    df["device_os"] = metadata.apply(
        lambda x: x.get("device_info", {}).get("os")
    )

    df["ip_address"] = metadata.apply(
        lambda x: x.get("device_info", {}).get("ip_address")
    )

    df["payment_method"] = metadata.apply(
        lambda x: x.get("payment_method")
    )

    df["risk_score"] = pd.to_numeric(
        metadata.apply(
            lambda x: x.get("risk_indicators", {}).get("risk_score")
        ),
        errors="coerce",
    )

    df["customer_risk_score"] = pd.to_numeric(
        metadata.apply(
            lambda x: x.get("risk_indicators", {}).get(
                "customer_risk_score"
            )
        ),
        errors="coerce",
    )

    df["amount_vs_average"] = pd.to_numeric(
        metadata.apply(
            lambda x: x.get("risk_indicators", {}).get(
                "amount_vs_average"
            )
        ),
        errors="coerce",
    )

    df["sender_degree_metadata"] = pd.to_numeric(
        metadata.apply(
            lambda x: x.get("network_metrics", {}).get(
                "sender_degree"
            )
        ),
        errors="coerce",
    )

    df["sender_clustering_metadata"] = pd.to_numeric(
        metadata.apply(
            lambda x: x.get("network_metrics", {}).get(
                "sender_clustering"
            )
        ),
        errors="coerce",
    )

    # Keep parsed metadata for later use if needed
    df["metadata_parsed"] = metadata

    return df.reset_index(drop=True)
