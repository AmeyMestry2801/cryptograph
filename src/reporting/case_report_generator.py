import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
from datetime import datetime

def generate_case_report(cluster_ids, test_df, run_metrics):
    cluster_df = test_df[test_df["txId"].isin(cluster_ids)].copy()

    if cluster_df.empty:
        return "ERROR: No matching transactions found for this cluster in test_df."

    avg_indeg = cluster_df["in_degree"].mean()
    avg_outdeg = cluster_df["out_degree"].mean()
    avg_pagerank = cluster_df["pagerank"].mean()

    # Simple, honest pattern classification based on degree ratio —
    # NOT a validated peeling-chain detector (that needs its own stage);
    # this is a first-pass structural description.
    if avg_outdeg > avg_indeg * 1.5:
        pattern_note = ("Nodes show higher out-degree than in-degree on average, "
                         "consistent with fan-out/distribution structure rather than "
                         "a simple linear chain. Further validation needed before "
                         "labeling as a specific typology.")
    elif avg_indeg > avg_outdeg * 1.5:
        pattern_note = ("Nodes show higher in-degree than out-degree, consistent with "
                         "a convergence/aggregation structure (multiple sources funneling "
                         "into fewer downstream transactions).")
    else:
        pattern_note = ("In-degree and out-degree are roughly balanced across the cluster; "
                         "no strong fan-in or fan-out signature detected in this snapshot.")

    report = f"""
CASE REPORT — Flagged Transaction Cluster
Generated: {datetime.now().isoformat()}
==========================================

CLUSTER OVERVIEW
- Transactions in cluster: {len(cluster_df)}
- Transaction IDs: {sorted(cluster_ids)}

STRUCTURAL EVIDENCE (computed independently on test-split graph, no leakage)
- Average in-degree:  {avg_indeg:.2f}
- Average out-degree: {avg_outdeg:.2f}
- Average PageRank:   {avg_pagerank:.6f}

INVESTIGATIVE NOTE
{pattern_note}

MODEL CONTEXT (for analyst reference — not a determination of guilt)
- Model precision on held-out test set: {run_metrics['precision']:.2%}
- Model recall on held-out test set:    {run_metrics['recall']:.2%}
- Interpretation: when this model flags a transaction, it is correct
  roughly {run_metrics['precision']:.0%} of the time on unseen data;
  it misses roughly {(1-run_metrics['recall']):.0%} of true illicit
  transactions in this evaluation window. This report should be treated
  as an investigative lead requiring human review, not a verdict.

LIMITATIONS
- Cluster derived from co-flagged connectivity within a single time window
  (test steps 35-49); chains spanning the train/test boundary are not
  captured by this analysis.
- Underlying Elliptic features are anonymized/aggregated; no raw
  transaction amounts are available, so amount-based patterns
  (e.g. classic decreasing-output peeling chains) cannot be verified here.
==========================================
"""
    return report


if __name__ == "__main__":
    with open("reports/metrics/largest_flagged_cluster.json") as f:
        cluster_ids = json.load(f)
    with open("reports/metrics/latest_run_metrics.json") as f:
        run_metrics = json.load(f)

    # Reload test_df the same way train.py built it, so features match
    from ingestion.load_elliptic import load_raw
    from splitting.time_split import split_by_time
    from graph_features.build_features import compute_topology_features
    from splitting.time_split import split_edgelist_by_node_membership

    features, classes, edgelist = load_raw()
    classes.columns = ["txId", "label"]
    df = features.merge(classes, on="txId", how="left")
    train_df, test_df = split_by_time(df)
    train_ids, test_ids = set(train_df["txId"]), set(test_df["txId"])
    _, test_edges, _ = split_edgelist_by_node_membership(edgelist, train_ids, test_ids)
    test_topo = compute_topology_features(test_edges, test_ids)
    test_df = test_df.merge(test_topo, on="txId", how="left")

    report = generate_case_report(cluster_ids, test_df, run_metrics)
    print(report)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/sample_case_report.txt", "w") as f:
        f.write(report)
    print("Saved to reports/sample_case_report.txt")