"""
Time-based train/test split for the Elliptic dataset.
Train: time steps 1-34. Test: time steps 35-49.
This matches the standard Elliptic benchmark split and avoids
leaking future graph structure into training.

IMPORTANT: this module only splits the transaction/label tables.
Graph objects (PyG Data, NetworkX graphs) must be built SEPARATELY
per split, downstream of this — never build one graph on the full
dataset and then split it.
"""
import pandas as pd

TRAIN_MAX_STEP = 34

def split_by_time(df: pd.DataFrame, time_col: str = "time_step"):
    train_df = df[df[time_col] <= TRAIN_MAX_STEP].copy()
    test_df = df[df[time_col] > TRAIN_MAX_STEP].copy()
    return train_df, test_df


def split_edgelist_by_node_membership(edgelist: pd.DataFrame, train_ids: set, test_ids: set):
    """
    An edge only belongs to the train graph if BOTH endpoints are train nodes,
    and only to the test graph if BOTH endpoints are test nodes.
    Cross-split edges (train node -> test node) are dropped from both —
    this is a deliberate, documented choice: including them would either
    leak test-side structure into train features or vice versa.
    """
    src_col, dst_col = edgelist.columns[0], edgelist.columns[1]

    train_edges = edgelist[
        edgelist[src_col].isin(train_ids) & edgelist[dst_col].isin(train_ids)
    ].copy()
    test_edges = edgelist[
        edgelist[src_col].isin(test_ids) & edgelist[dst_col].isin(test_ids)
    ].copy()
    cross_edges = len(edgelist) - len(train_edges) - len(test_edges)

    return train_edges, test_edges, cross_edges

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # points to src/
    from ingestion.load_elliptic import load_raw
# if __name__ == "__main__":
#     import sys
#     sys.path.append("..")
#     from ingestion.load_elliptic import load_raw

    features, classes, edgelist = load_raw()
    classes.columns = ["txId", "label"]
    df = features.merge(classes, on="txId", how="left")

    train_df, test_df = split_by_time(df)
    train_ids = set(train_df["txId"])
    test_ids = set(test_df["txId"])

    train_edges, test_edges, cross_edges = split_edgelist_by_node_membership(
        edgelist, train_ids, test_ids
    )

    print(f"Train nodes: {len(train_ids):,}  Test nodes: {len(test_ids):,}")
    print(f"Train edges: {len(train_edges):,}  Test edges: {len(test_edges):,}")
    print(f"Cross-split edges dropped: {cross_edges:,} "
          f"({cross_edges/len(edgelist)*100:.2f}% of total)")