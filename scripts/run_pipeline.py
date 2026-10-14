"""Run the full segmentation pipeline from the command line (no Streamlit or Plotly needed).

Examples
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --data my_customers.csv --algorithm dbscan
    python scripts/run_pipeline.py --algorithm hierarchical --k 4 --linkage average

Writes to outputs/: segmented_customers.csv, segment_profiles.csv,
cluster_scan.csv and segment_insights.md.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.clustering import (default_dbscan_grid, default_min_samples, dbscan_grid_search,  # noqa: E402
                                     run_dbscan, run_hierarchical, run_kmeans, suggest_dbscan_params, suggest_eps)
from segmentation.config import DEFAULT_FEATURES, K_MAX, K_MIN, OUTPUT_DIR  # noqa: E402
from segmentation.data_loader import detect_id_column, load_csv, load_sample, numeric_candidates  # noqa: E402
from segmentation.evaluation import cluster_scan, evaluate_labels, recommend_k  # noqa: E402
from segmentation.insights import (build_insights, insights_to_markdown, profile_segments,  # noqa: E402
                                   profile_table, segment_name_map)
from segmentation.preprocessing import SCALERS, PreprocessConfig, preprocess  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, help="CSV file (default: bundled sample)")
    p.add_argument("--algorithm", choices=["kmeans", "hierarchical", "dbscan"], default="kmeans")
    p.add_argument("--k", type=int, help="clusters for kmeans/hierarchical (default: recommended)")
    p.add_argument("--linkage", default="ward", choices=["ward", "complete", "average", "single"])
    p.add_argument("--eps", type=float, help="DBSCAN eps (default: from parameter search)")
    p.add_argument("--min-samples", type=int, help="DBSCAN min_samples")
    p.add_argument("--scaler", choices=list(SCALERS), default="standard")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    df = load_csv(args.data) if args.data else load_sample()
    id_col = detect_id_column(df)
    numeric = numeric_candidates(df, id_col)
    features = [c for c in DEFAULT_FEATURES if c in numeric] or numeric[:6]

    pre = preprocess(df, PreprocessConfig(numeric_features=features, scaler=args.scaler, id_column=id_col))
    X = pre.X
    print(f"Customers: {len(X):,} | features: {', '.join(features)}")
    print(f"Preprocessing: {sum(pre.report['missing_before'].values())} missing cells imputed, "
          f"{sum(pre.report['clipped_values'].values())} outlier values clipped, scaler={args.scaler}")

    scan = cluster_scan(X, range(K_MIN, min(K_MAX, len(X) - 1) + 1))
    rec = recommend_k(scan)
    print(f"Elbow k={rec['elbow_k']} | best silhouette k={rec['silhouette_k']} | recommended k={rec['recommended_k']}")

    k = args.k or rec["recommended_k"]
    if args.algorithm == "kmeans":
        result = run_kmeans(X, k)
    elif args.algorithm == "hierarchical":
        result = run_hierarchical(X, k, args.linkage)
    else:
        ms = args.min_samples or default_min_samples(X.shape[1])
        eps = args.eps
        if eps is None:
            grid = dbscan_grid_search(X, *default_dbscan_grid(X, ms))
            suggestion = suggest_dbscan_params(grid)
            eps, ms = suggestion if suggestion else (suggest_eps(X, ms), ms)
        result = run_dbscan(X, eps, ms)
    print(f"{result.algorithm}: {result.params}")

    metrics = evaluate_labels(X, result.labels)
    print(f"Segments: {metrics['clusters']} | silhouette: {metrics['silhouette']:.3f} | "
          f"Davies-Bouldin: {metrics['davies_bouldin']:.3f} | noise: {metrics['noise_pct']:.1f}%")
    if result.n_clusters < 1:
        sys.exit("No clusters found; adjust the parameters.")

    profile = profile_segments(pre.data, result.labels, id_col)
    insights = build_insights(profile, features)
    names = segment_name_map(insights)

    args.out.mkdir(parents=True, exist_ok=True)
    seg = pre.data.copy()
    seg["segment_id"] = result.labels
    seg["segment"] = [names[int(c)] for c in result.labels]
    seg.to_csv(args.out / "segmented_customers.csv", index=False)
    profile_table(profile, insights).to_csv(args.out / "segment_profiles.csv", index=False)
    scan.to_csv(args.out / "cluster_scan.csv", index=False)
    (args.out / "segment_insights.md").write_text(insights_to_markdown(
        insights, context={"Algorithm": result.algorithm, "Customers": f"{len(X):,}",
                           "Silhouette": f"{metrics['silhouette']:.3f}"}), encoding="utf-8")

    print("\nSegments")
    for ins in insights:
        spend = f", {ins.spend_share_pct:.0f}% of est. spend" if ins.spend_share_pct is not None else ""
        print(f"  {ins.name}: {ins.customers:,} customers ({ins.share_pct:.1f}%{spend})")
    print(f"\nFiles written to {args.out}")


if __name__ == "__main__":
    main()
