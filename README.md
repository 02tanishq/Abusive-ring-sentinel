# Abuse Ring Sentinel

A graph-based AML risk detection system designed for the **AI Risk Manager — Track 02: Stop the merchant losing money to fraud, returns and chargebacks** challenge.

The project focuses on the **Abuse-ring sentinel** direction: detecting coordinated suspicious transaction structures by combining transaction-level risk, account-level risk, graph topology, and behavioral signals.

The goal is not just to produce a fraud score. It is to identify **high-risk groups of connected accounts and transactions** that deserve investigation while explicitly measuring false positives and their operational cost.

---

## Architecture

```text
                         ┌─────────────────────┐
                         │     RAW DATASET     │
                         │                     │
                         │ transactions        │
                         │ accounts            │
                         │ timestamps          │
                         │ amounts             │
                         │ metadata            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FEATURE ENGINEERING │
                         │                     │
                         │ Transaction         │
                         │ Account / Node      │
                         │ Sender Graph        │
                         │ Temporal / Amount   │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
      ┌─────────────────────┐               ┌─────────────────────┐
      │   NETWORKX GRAPH    │               │      XGBOOST        │
      │                     │               │                     │
      │ Nodes = accounts    │               │ Transaction-level   │
      │ Edges = transactions│               │ features            │
      │                     │               │                     │
      │ Master graph        │               │ → Transaction risk  │
      └──────────┬──────────┘               └──────────┬──────────┘
                 │                                     │
                 ▼                                     │
      ┌─────────────────────┐                          │
      │        PyG          │                          │
      │                     │                          │
      │ Graph → tensors     │                          │
      │ node features       │                          │
      │ edge features       │                          │
      └──────────┬──────────┘                          │
                 │                                     │
                 ▼                                     │
      ┌─────────────────────┐                          │
      │       GNN           │                          │
      │     GraphSAGE       │                          │
      │                     │                          │
      │ → Node embeddings   │                          │
      └──────────┬──────────┘                          │
                 │                                     │
                 ▼                                     │
      ┌─────────────────────┐                          │
      │        MLP          │                          │
      │                     │                          │
      │ Node features +     │                          │
      │ GNN embeddings      │                          │
      │                     │                          │
      │ → Node risk score   │                          │
      └──────────┬──────────┘                          │
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ TOPOLOGY DETECTION  │
                         │                     │
                         │ NetworkX graph      │
                         │ + Node risk score   │
                         │ + Transaction risk  │
                         │                     │
                         │ • Fan-in            │
                         │ • Fan-out           │
                         │ • Reciprocal        │
                         │ • 3–6 node cycles   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  DEEP RISK ANALYSIS │
                         │                     │
                         │ Each topology       │
                         │ candidate analyzed  │
                         │ independently       │
                         │                     │
                         │ Node risk            │
                         │ + Transaction risk  │
                         │ + Topology strength │
                         │ + Temporal behavior │
                         │ + Amount behavior   │
                         │ + Structure         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ FINAL RING RISK     │
                         │ SCORE               │
                         │                     │
                         │ → Alert / No Alert  │
                         └─────────────────────┘
```

## Model data flow

The two predictive models are separate branches from feature engineering:

```text
                       FEATURE ENGINEERING
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Graph / Node Features       Transaction Features
                │                           │
                ▼                           ▼
           NetworkX → PyG               XGBoost
                │                           │
                ▼                           ▼
            GraphSAGE             Transaction Risk Score
                │
                ▼
               MLP
                │
                ▼
          Node Risk Score
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                    TOPOLOGY DETECTION
                              ▲
                              │
                       NetworkX Graph
                              │
                              ▼
                    DEEP RISK ANALYSIS
                              │
                              ▼
                       Ring Risk Score
```

**Important:** XGBoost receives transaction features directly. It does **not** take the node risk score as an input. The topology stage receives the NetworkX master graph together with both independently produced risk scores.

## Core idea

The system uses **multiple levels of evidence** rather than relying on a single classifier.

```text
Transaction level
        │
        └── "How risky is this transaction?"

Account level
        │
        └── "How risky is this account?"

Graph topology
        │
        └── "How are these accounts connected?"

Behavioral analysis
        │
        └── "Does the group behave like a coordinated pattern?"

                    ↓

              Final Ring Risk
                    ↓
                  Alert
```

This separation is intentional:

- The **master graph** represents transaction relationships.
- The **XGBoost model** provides transaction-level risk.
- **GraphSAGE + MLP** provide account-level risk.
- **Topology detection** identifies candidate structures.
- **Deep risk analysis** combines the available signals for each topology candidate.
- Ground truth remains in the transaction dataset rather than being stored on graph edges.

---

## Pipeline

### 1. Data ingestion

The raw transaction dataset is cleaned and normalized.

A stable `transaction_id` is assigned once and is used throughout the pipeline to connect:

- raw transactions
- graph edges
- transaction features
- transaction risk scores
- evaluation ground truth

The graph itself does not store the target label.

---

### 2. Feature engineering

The project creates separate features for different modeling levels.

#### Transaction features

Examples include:

- amount
- log amount
- hour and day cyclic features
- transaction type
- payment method
- time since previous transaction
- previous transaction amount
- amount-vs-previous behavior
- rapid activity
- structuring / amount similarity signals
- sender structural features

#### Account features

Examples include:

- incoming transaction count
- outgoing transaction count
- total incoming amount
- total outgoing amount
- average / minimum / maximum transaction amount
- total activity
- net flow
- flow imbalance
- unique counterparties
- IP-related features

#### Sender graph features

Examples include:

- PageRank
- hub score
- in-degree
- out-degree
- inbound/outbound ratio
- flow asymmetry

---

## 3. Master NetworkX graph

A single master graph is constructed:

```text
Node  = account
Edge  = transaction
```

The graph is a `networkx.MultiDiGraph`.

Multiple transactions between the same pair of accounts are retained as separate edges using `transaction_id`.

Example:

```text
C4638 ── transaction 0 ─────▶ C1811
C4638 ── transaction 55746 ─▶ C1811
C4638 ── transaction 70649 ─▶ C1811
```

This preserves transaction-level structure while allowing the topology stage to reason about account relationships.

The master graph is built once and reused.

---

## 4. Graph neural network

The graph is converted to PyTorch Geometric tensors.

The current node model uses **GraphSAGE** to learn structural account representations.

```text
Account features
       +
Graph structure
       ↓
   GraphSAGE
       ↓
Node embeddings
```

The learned embedding is then combined with the engineered account features.

An MLP predicts the account-level risk score.

```text
Raw account features
        +
GraphSAGE embedding
        ↓
       MLP
        ↓
   Node risk score
```

---

## 5. Transaction risk model

A separate **XGBoost** model predicts transaction-level risk using transaction features.

The transaction model and node-risk model serve different purposes:

```text
XGBoost
   ↓
Transaction risk

GraphSAGE + MLP
   ↓
Account risk
```

The fitted transaction model also produces an all-transaction risk artifact:

```text
artifacts/transaction_risk_all.parquet
```

This is used later during topology/deep analysis so that every topology candidate can access its transaction risk.

---

## 6. Topology detection

Topology detection is performed after risk estimation.

A low risk threshold can be used at the topology stage as a **compute-pruning mechanism** so that the full 1.09M-edge graph does not require expensive pattern searches everywhere.

The topology detector looks for structural patterns such as:

```text
Fan-in

A ──▶ X
B ──▶ X
C ──▶ X
D ──▶ X
```

```text
Fan-out

X ──▶ A
X ──▶ B
X ──▶ C
X ──▶ D
```

```text
Reciprocal

A ⇄ B
```

```text
3-cycle

A ─▶ B
▲    │
│    ▼
C ◀──┘
```

and bounded cycles from **3 to 6 accounts**.

The topology stage generates candidate subgraphs for deeper analysis.

---

## 7. Deep risk analysis

Each topology candidate is analyzed **independently**.

There is deliberately:

```text
NO candidate merging
NO overlap-based ring expansion
NO post-hoc graph construction
```

The analysis only uses the accounts and transactions belonging to that topology candidate.

The candidate receives several signals.

### Node risk

```text
Mean node risk
Maximum node risk
Fraction of high-risk accounts
```

Combined as:

```text
N =
    0.50 × mean node risk
  + 0.30 × maximum node risk
  + 0.20 × high-risk-account fraction
```

### Transaction risk

```text
Mean transaction risk
Maximum transaction risk
Fraction of high-risk transactions
```

Combined as:

```text
T =
    0.40 × mean transaction risk
  + 0.40 × maximum transaction risk
  + 0.20 × high-risk-transaction fraction
```

### Topology strength

Current structural priors are:

```text
3-cycle        1.00
4-cycle        0.95
5-cycle        0.90
6-cycle        0.85
reciprocal     0.30
fan-in/out     0.30–0.50 depending on size
```

These values are used as structural signal, not as the ground truth.

### Temporal behavior

The analysis considers:

- rapid activity
- layering time span
- placement time span

### Amount behavior

The analysis considers:

- amount similarity
- pass-through behavior

### Additional diagnostics

The candidate also records:

- repeated counterparties
- risk sequence behavior
- time span
- highest-risk accounts
- highest-risk transactions

---

## 8. Final ring-risk score

The current score is:

```text
Ring Risk Score =

    0.25 × Node Risk
  + 0.25 × Transaction Risk
  + 0.15 × Topology Strength
  + 0.15 × Temporal Score
  + 0.10 × Amount Score
  + 0.10 × Structural Score
```

The current diagnostic alert threshold is:

```text
Ring Risk Score >= 0.50
```

The threshold is treated as an operating point and should be evaluated against the precision/recall/cost trade-off rather than selected only by F1.

---

## Ground truth

The project uses `isFraud` as the single target label.

For the current AMLNet data, `isFraud` and `isMoneyLaundering` were verified to be identical for all transactions:

```text
Mismatch count: 0
```

Therefore:

```text
transaction_id
      ↓
clean_transactions.parquet
      ↓
isFraud
```

The master graph does **not** need to store the target label.

This keeps responsibilities separate:

```text
Graph
  → structure

Risk models
  → predicted risk

Transaction dataset
  → ground truth
```

---

## Current result

At the current diagnostic threshold of `0.50`, the deep-risk stage produced:

```text
Topology candidates:       14,930

Predicted alerts:             159
True-positive alerts:         155
False-positive alerts:          4
False-negative alerts:        912
True-negative candidates:  13,859
```

Metrics:

```text
Alert precision:        97.48%
Alert recall:           14.53%
False-alert rate:        2.52%
Classical FPR:           0.03%
False-positive cost:   ₹20,000
```

The current system therefore behaves as a **high-confidence alerting system**: it produces very few false alerts, but the current threshold sacrifices recall.

These numbers should be treated as the current operating point, not as the final optimized model.

---

## Project structure

```text
abuse-ring-sentinel/
│
├── src/
│   ├── ingest.py
│   ├── graph_builder.py
│   ├── feature_engineering.py
│   ├── topology.py
│   ├── deep_risk_analysis.py
│   │
│   └── models/
│       ├── node_embeddings.py
│       ├── node_risk.py
│       └── Transaction_risk_score.py
│
├── artifacts/
│   ├── clean_transactions.parquet
│   ├── master_graph.pkl
│   ├── node_risk_results.parquet
│   ├── transaction_risk_all.parquet
│   ├── topology_patterns.parquet
│   └── ring_analysis.parquet
│
├── pipeline.py
├── requirements.txt
└── README.md
```

---

## Main artifacts

### `master_graph.pkl`

The complete NetworkX transaction graph.

```text
11,000 nodes
1,090,000 transaction edges
```

Each edge retains its:

```text
transaction_id
```

---

### `node_risk_results.parquet`

Account-level risk produced by:

```text
GraphSAGE
    +
account features
    ↓
MLP
```

Contains:

```text
account_id
risk_score
```

---

### `transaction_risk_all.parquet`

Transaction-level risk produced by the fitted XGBoost model.

Contains risk scores for the full transaction dataset.

---

### `topology_patterns.parquet`

Topology candidates generated by the structural detection stage.

Examples:

```text
fan_in
fan_out
reciprocal
3_cycle
4_cycle
5_cycle
6_cycle
```

---

### `ring_analysis.parquet`

Final deep-analysis results for every topology candidate.

It contains:

```text
candidate_id
pattern_type
num_accounts
num_transactions

node-risk signals
transaction-risk signals
topology signals
temporal signals
amount signals
structural signals

ring_score
predicted_ring_alert

ground-truth counts
alert_class
```

---

## Running the project

Run the complete pipeline:

```bash
python pipeline.py
```

Or run deep analysis after the prerequisite artifacts have been generated:

```bash
python src/deep_risk_analysis.py
```

The deep-analysis stage expects:

```text
artifacts/master_graph.pkl
artifacts/clean_transactions.parquet
artifacts/node_risk_results.parquet
artifacts/transaction_risk_all.parquet
```

and the topology output parquet.

---

## Design principles

### 1. Multi-level evidence

No single signal is expected to explain a suspicious ring.

```text
Transaction risk
      +
Account risk
      +
Topology
      +
Time
      +
Amount behavior
      +
Structural behavior
```

produces the final risk score.

### 2. Preserve transaction identity

`transaction_id` is the link between:

```text
dataset
graph
transaction model
topology
deep analysis
ground truth
```

### 3. Keep the graph structural

The graph is not used as a storage location for every model target or prediction.

This keeps:

```text
structure
```

separate from:

```text
prediction
```

and:

```text
ground truth
```

### 4. Analyze topology candidates independently

The deep stage does not merge nearby candidates into larger rings.

Each topology result is treated as its own investigative unit.

### 5. Measure investigator cost

A detector that generates many false alerts can be unusable even with good recall.

Therefore the project explicitly tracks:

```text
false alerts
false-alert rate
false-positive rate
false-positive cost
```

---

## Future evaluation

The next evaluation stage should focus on the operating trade-off between recall and false-positive cost.

Recommended threshold sweep:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
```

For every threshold, measure:

```text
alerts
precision
recall
false alerts
false-alert rate
false-positive cost
```

Evaluation should also distinguish between:

```text
transaction-level coverage
candidate/ring-level coverage
account-level coverage
```

This makes it possible to determine whether the main limitation is:

```text
the ring-risk threshold
```

or:

```text
the topology stage failing to generate relevant candidates
```

---

## Track alignment

**Track:** AI Risk Manager — Track 02

**Direction:** Abuse-ring sentinel

**Objective:** Build a working risk detector with honest held-out evaluation and explicit false-positive cost.

The system is designed as a defensive risk-management pipeline:

```text
Detect
   ↓
Rank
   ↓
Explain
   ↓
Alert
   ↓
Measure investigation cost
```

It is not designed to facilitate fraud or bypass financial controls.
