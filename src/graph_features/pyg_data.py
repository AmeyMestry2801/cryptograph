import torch
from torch_geometric.data import Data
import pandas as pd

def build_pyg_data(df: pd.DataFrame, edge_df: pd.DataFrame, feature_cols: list) -> Data:
    node_ids = df["txId"].tolist()
    id_to_idx = {tx: i for i, tx in enumerate(node_ids)}

    x = torch.tensor(df[feature_cols].values, dtype=torch.float)

    label_map = {"illicit": 1, "licit": 0, "unknown": -1}
    y = torch.tensor(df["label"].map(label_map).values, dtype=torch.long)
    mask = y != -1  # labeled nodes only

    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]
    edges = edge_df[[src_col, dst_col]].values
    edge_index = torch.tensor(
        [[id_to_idx[s] for s, d in edges], [id_to_idx[d] for s, d in edges]],
        dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.mask = mask
    return data