"""
Degree + PageRank, computed SEPARATELY per split.
Never call this on the full graph before splitting.
"""
import networkx as nx
import pandas as pd

def compute_topology_features(edge_df: pd.DataFrame, node_ids: set) -> pd.DataFrame:
    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]
    G = nx.from_pandas_edgelist(edge_df, src_col, dst_col, create_using=nx.DiGraph())
    G.add_nodes_from(node_ids)  # include isolated nodes too

    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    pr = nx.pagerank(G, alpha=0.85, max_iter=200)

    out = pd.DataFrame({
        "txId": list(node_ids),
    })
    out["in_degree"] = out["txId"].map(indeg).fillna(0)
    out["out_degree"] = out["txId"].map(outdeg).fillna(0)
    out["pagerank"] = out["txId"].map(pr).fillna(0)
    return out