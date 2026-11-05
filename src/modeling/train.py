import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from ingestion.load_elliptic import load_raw
from splitting.time_split import split_by_time, split_edgelist_by_node_membership
from graph_features.build_features import compute_topology_features
from graph_features.pyg_data import build_pyg_data
from modeling.gnn_model import GCN

torch.manual_seed(42)

print("Loading raw data...")
features, classes, edgelist = load_raw()
classes.columns = ["txId", "label"]
df = features.merge(classes, on="txId", how="left")
label_map = {"1": "illicit", "2": "licit", "unknown": "unknown"}
df["label"] = df["label"].astype(str).map(label_map)

print("Splitting by time step (train<=34, test>34)...")
train_df, test_df = split_by_time(df)
train_ids, test_ids = set(train_df["txId"]), set(test_df["txId"])
train_edges, test_edges, cross = split_edgelist_by_node_membership(edgelist, train_ids, test_ids)
print(f"Cross-split edges dropped: {cross}")

print("Computing graph features SEPARATELY per split...")
train_topo = compute_topology_features(train_edges, train_ids)
test_topo = compute_topology_features(test_edges, test_ids)

train_df = train_df.merge(train_topo, on="txId", how="left")
test_df = test_df.merge(test_topo, on="txId", how="left")

feat_cols = [c for c in train_df.columns if c.startswith("feat_")] + ["in_degree", "out_degree", "pagerank"]

print("Building PyG Data objects...")
train_data = build_pyg_data(train_df, train_edges, feat_cols)
test_data = build_pyg_data(test_df, test_edges, feat_cols)

print(f"Train graph: {train_data.x.shape[0]} nodes, {train_data.mask.sum().item()} labeled")
print(f"Test graph:  {test_data.x.shape[0]} nodes, {test_data.mask.sum().item()} labeled")

model = GCN(in_channels=train_data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training (unweighted loss — kept as reported baseline; class-weighted "
      "variant was tested and traded 28pts of precision for 7pts of recall, "
      "net negative on F1/AUPRC)...")
model.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = F.cross_entropy(out[train_data.mask], train_data.y[train_data.mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

print("\nEvaluating on TEST set (steps 35-49, never seen in training)...")
model.eval()
with torch.no_grad():
    out = model(test_data.x, test_data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)

y_true = test_data.y[test_data.mask].numpy()
y_pred = preds[test_data.mask].numpy()
y_prob = probs[test_data.mask].numpy()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auprc = average_precision_score(y_true, y_prob)

print("\n=== HONEST TEST-SET METRICS (post-split-fix, unweighted, reported baseline) ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"AUPRC:     {auprc:.4f}")
print(f"Compare to V1's reported 79% recall (unverified split methodology).")

# ============================================================
# STAGE: Cluster flagged nodes into topology-based groups
# ============================================================
print("\n=== Building subgraph of model-flagged nodes (test set) ===")

test_labeled_df = test_df.loc[test_data.mask.numpy()].reset_index(drop=True)
flagged_ids = test_labeled_df.loc[y_pred == 1, "txId"].tolist()
print(f"Model flagged {len(flagged_ids)} transactions as illicit in test set.")

G_test = nx.from_pandas_edgelist(
    test_edges, test_edges.columns[0], test_edges.columns[1], create_using=nx.DiGraph()
)
G_flagged = G_test.subgraph(flagged_ids).copy()

components = list(nx.weakly_connected_components(G_flagged))
components = [c for c in components if len(c) >= 3]
components.sort(key=len, reverse=True)

print(f"Found {len(components)} flagged clusters with 3+ connected transactions.")
if components:
    print(f"Largest cluster size: {len(components[0])} transactions.")
    print(f"Cluster size distribution (top 5): {[len(c) for c in components[:5]]}")
else:
    print("No multi-node flagged clusters found in this run.")

# Save artifacts for the next stage (case report generator) so we don't
# have to retrain every time we iterate on report formatting.
import json
Path("reports/metrics").mkdir(parents=True, exist_ok=True)

with open("reports/metrics/latest_run_metrics.json", "w") as f:
    json.dump({
        "precision": precision, "recall": recall, "f1": f1, "auprc": auprc,
        "n_flagged": len(flagged_ids), "n_clusters": len(components),
        "largest_cluster_size": len(components[0]) if components else 0,
    }, f, indent=2)

if components:
    with open("reports/metrics/largest_flagged_cluster.json", "w") as f:
        json.dump(list(components[0]), f, indent=2)

print("\nSaved run metrics and largest cluster to reports/metrics/")