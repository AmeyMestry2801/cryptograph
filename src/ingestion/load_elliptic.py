"""
Elliptic dataset ingestion.
Loads raw CSVs, builds a canonical graph object, reports REAL counts.
No train/test split happens here — that's a separate stage (src/splitting/)
so we never accidentally compute anything graph-derived before the split.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_raw():
    # Elliptic's raw features/classes files ship with NO header row.
    # If we don't pass header=None explicitly, pandas treats row 0 as
    # column names silently — wrong dtypes, one row missing, no error.
    features = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")  # this one DOES have a header
    edgelist = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")  # this one DOES have a header

    # Name feature columns explicitly: col 0 = txId, col 1 = time step, rest = 165 features
    feature_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, features.shape[1] - 1)]
    features.columns = feature_cols

    return features, classes, edgelist


def sanity_report(features, classes, edgelist):
    print("=== RAW SHAPE CHECK ===")
    print(f"features.csv rows: {len(features):,}  cols: {features.shape[1]}")
    print(f"classes.csv rows:  {len(classes):,}")
    print(f"edgelist.csv rows (directed edges): {len(edgelist):,}")

    print("\n=== CLASS LABEL DISTRIBUTION (raw, unmapped) ===")
    print(classes.iloc[:, 1].value_counts())

    print("\n=== TIME STEP RANGE ===")
    print(f"min: {features['time_step'].min()}  max: {features['time_step'].max()}  "
          f"unique steps: {features['time_step'].nunique()}")

    print("\n=== NODE/EDGE COUNT ===")
    n_nodes = features["txId"].nunique()
    n_edges = len(edgelist)
    print(f"Unique transaction nodes: {n_nodes:,}")
    print(f"Directed edges: {n_edges:,}")

    # Confirm every txId in the edgelist actually exists in features —
    # if not, we have orphan edges pointing to nodes with no feature vector.
    node_ids = set(features["txId"])
    edge_ids = set(edgelist.iloc[:, 0]) | set(edgelist.iloc[:, 1])
    orphans = edge_ids - node_ids
    print(f"Edge endpoints with NO matching feature row (orphans): {len(orphans):,}")


if __name__ == "__main__":
    features, classes, edgelist = load_raw()
    sanity_report(features, classes, edgelist)