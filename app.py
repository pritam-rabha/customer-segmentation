"""Customer Segmentation dashboard.

Run with:  streamlit run app.py

The UI only orchestrates; all analytics live in the ``segmentation`` package.
"""
from __future__ import annotations

import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run from any working directory

import numpy as np
import pandas as pd
import streamlit as st

from segmentation import visualization as viz
from segmentation.clustering import (LINKAGES, compute_linkage_matrix, dbscan_grid_search, default_dbscan_grid,
                                     default_min_samples, k_distance_curve, run_dbscan, run_hierarchical,
                                     run_kmeans, suggest_dbscan_params, suggest_eps)
from segmentation.config import DEFAULT_FEATURES, K_MAX, K_MIN, MAX_HIERARCHICAL_ROWS
from segmentation.data_loader import (DataValidationError, categorical_candidates, detect_id_column,
                                      load_csv, load_sample, numeric_candidates)
from segmentation.evaluation import (agreement_matrix, cluster_scan, evaluate_labels, find_elbow_k,
                                     recommend_k, silhouette_by_sample)
from segmentation.insights import (build_insights, insights_to_markdown, profile_segments, profile_table,
                                   segment_name_map)
from segmentation.preprocessing import SCALERS, PreprocessConfig, preprocess
from segmentation.formatting import format_inr
from segmentation.schema import display_name

st.set_page_config(page_title="Customer segmentation", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"<style>{(Path(__file__).parent / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

KMEANS, HIER, DBSCAN_ = "K-Means", "Hierarchical", "DBSCAN"
MISSING_LABELS = {"median": "Fill with median", "mean": "Fill with mean", "drop": "Drop rows"}
OUTLIER_LABELS = {"iqr_clip": "Clip to 1.5 x IQR fences", "none": "Keep as is"}
SCALER_LABELS = {"standard": "Standardise (z-score)", "minmax": "Min-max normalise (0-1)", "robust": "Robust (median / IQR)"}


# ----------------------------------------------------------------------- caching
@st.cache_data(show_spinner=False)
def cached_sample() -> pd.DataFrame:
    return load_sample()


@st.cache_data(show_spinner=False)
def cached_upload(raw: bytes) -> pd.DataFrame:
    return load_csv(io.BytesIO(raw))


@st.cache_data(show_spinner="Preprocessing data")
def cached_preprocess(df: pd.DataFrame, cfg: dict):
    return preprocess(df, PreprocessConfig(**cfg))


@st.cache_data(show_spinner="Scanning cluster counts")
def cached_scan(X: np.ndarray, k_max: int, algorithm: str, linkage: str) -> pd.DataFrame:
    return cluster_scan(X, range(K_MIN, k_max + 1), algorithm, linkage)


@st.cache_data(show_spinner="Searching DBSCAN settings")
def cached_grid(X: np.ndarray, min_samples: int) -> pd.DataFrame:
    eps_values, ms_values = default_dbscan_grid(X, min_samples)
    return dbscan_grid_search(X, eps_values, ms_values)


@st.cache_data(show_spinner=False)
def cached_linkage(X: np.ndarray, method: str) -> np.ndarray:
    return compute_linkage_matrix(X, method)


@st.cache_data(show_spinner=False)
def cached_kdistance(X: np.ndarray, k: int) -> np.ndarray:
    return k_distance_curve(X, k)


@st.cache_data(show_spinner=False)
def cached_pca(X: np.ndarray, n: int):
    return viz.pca_projection(X, n)


@st.cache_data(show_spinner="Clustering")
def cached_cluster(X: np.ndarray, algorithm: str, k: int, linkage: str, eps: float, min_samples: int):
    if algorithm == KMEANS:
        return run_kmeans(X, k)
    if algorithm == HIER:
        return run_hierarchical(X, k, linkage)
    return run_dbscan(X, eps, min_samples)


def chart(fig, key: str) -> None:
    st.plotly_chart(fig, key=key)


# ---------------------------------------------------------------------- header
header_slot = st.empty()  # filled once the dataset and algorithm are known

# --------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.markdown('<div class="side-label">Dataset</div>', unsafe_allow_html=True)
source = sb.radio("Dataset", ["Bundled sample (1,000 customers)", "Upload a CSV"], label_visibility="collapsed")
try:
    if source.startswith("Upload"):
        upload = sb.file_uploader("CSV file", type="csv")
        if upload is None:
            st.info("Upload a customer CSV in the sidebar, or switch to the bundled sample. "
                    "The file needs one row per customer and at least two numeric columns.")
            st.stop()
        raw_df = cached_upload(upload.getvalue())
        dataset_name = upload.name
    else:
        raw_df = cached_sample()
        dataset_name = "Sample: 1,000 Indian retail customers"
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

id_col = detect_id_column(raw_df)
num_options = numeric_candidates(raw_df, id_col)
cat_options = categorical_candidates(raw_df, id_col)
default_feats = [c for c in DEFAULT_FEATURES if c in num_options] or num_options[:6]

sb.markdown('<div class="side-label">Features</div>', unsafe_allow_html=True)
features = sb.multiselect("Numeric features", num_options, default=default_feats, format_func=display_name)
cat_features = sb.multiselect("Categorical features (optional)", cat_options, format_func=display_name)
if len(features) < 2:
    st.warning("Select at least two numeric features in the sidebar. Categorical columns can be added on top of them, "
               "but customers described only by categories are mostly identical and cannot be clustered meaningfully.")
    st.stop()

with sb.expander("Preprocessing", expanded=False):
    missing_strategy = st.selectbox("Missing values", list(MISSING_LABELS), format_func=MISSING_LABELS.get)
    outlier_method = st.selectbox("Outliers", list(OUTLIER_LABELS), format_func=OUTLIER_LABELS.get)
    log_skewed = st.checkbox("Log-transform skewed columns", value=False)
    scaler = st.selectbox("Scaling", list(SCALERS), format_func=SCALER_LABELS.get)
    l2 = st.checkbox("L2-normalise each customer vector", value=False)
    drop_dups = st.checkbox("Remove duplicate customers", value=True)

try:
    pre = cached_preprocess(raw_df, dict(
        numeric_features=features, categorical_features=cat_features, missing_strategy=missing_strategy,
        drop_duplicates=drop_dups, outlier_method=outlier_method, log_transform_skewed=log_skewed,
        scaler=scaler, l2_normalize=l2, id_column=id_col))
except ValueError as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

X, data = pre.X, pre.data
n_rows, n_dims = X.shape
k_max = min(K_MAX, n_rows - 1)
kmeans_scan = cached_scan(X, k_max, "kmeans", "ward")
rec = recommend_k(kmeans_scan)

sb.markdown('<div class="side-label">Algorithm</div>', unsafe_allow_html=True)
algorithm = sb.selectbox("Method", [KMEANS, HIER, DBSCAN_], label_visibility="collapsed")
k, linkage, eps, min_samples = rec["recommended_k"], "ward", 0.5, default_min_samples(n_dims)

if algorithm in (KMEANS, HIER):
    k = sb.slider("Number of clusters (k)", K_MIN, k_max, rec["recommended_k"], key=f"k_{rec['recommended_k']}_{k_max}")
    sb.caption(f"Elbow suggests k = {rec['elbow_k']}; best silhouette at k = {rec['silhouette_k']}.")
if algorithm == HIER:
    linkage = sb.selectbox("Linkage", LINKAGES)
    if n_rows > MAX_HIERARCHICAL_ROWS:
        st.error(f"Hierarchical clustering is limited to {MAX_HIERARCHICAL_ROWS:,} customers (you have {n_rows:,}). "
                 "Choose K-Means or DBSCAN.")
        st.stop()
if algorithm == DBSCAN_:
    suggestion = suggest_dbscan_params(cached_grid(X, min_samples))
    eps_default, ms_default = suggestion if suggestion else (suggest_eps(X, min_samples), min_samples)
    min_samples = sb.slider("min_samples", 3, 60, int(ms_default), key=f"ms_{ms_default}_{n_dims}")
    eps_default = max(0.06, float(eps_default))
    eps = sb.slider("eps (neighbourhood radius)", 0.05, float(max(4.0, eps_default * 2)), eps_default,
                    step=0.01, key=f"eps_{eps_default}_{n_dims}")
    sb.caption("Defaults come from the parameter search in the 'Choose parameters' tab.")

# ------------------------------------------------------------ run + interpret
try:
    result = cached_cluster(X, algorithm, int(k), linkage, float(eps), int(min_samples))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

labels = result.labels
metrics = evaluate_labels(X, labels)
has_clusters = result.n_clusters >= 1

if has_clusters:
    profile = profile_segments(data, labels, id_col)
    insights = build_insights(profile, [c for c in pre.report["features_used"] if c in data.columns])
    names = segment_name_map(insights)
    segment_order = [names[i.cluster_id] for i in insights]
    plot_df = data.copy()
    plot_df["Segment"] = [names[int(c)] for c in labels]
    n_pc = min(3, n_dims)
    pcs, pc_var = cached_pca(X, n_pc)
    for i in range(n_pc):
        plot_df[f"PC{i + 1}"] = pcs[:, i]
else:
    insights = []

header_slot.markdown(
    f"""
    <div class="app-header">
      <div>
        <h1 class="app-title">Customer Segmentation</h1>
        <div class="app-sub">Behaviour-based clustering for Indian retail customers: find the groups, then decide what to do with each.</div>
      </div>
      <div class="chips">
        <span class="chip"><b>Data</b>{html.escape(dataset_name)}</span>
        <span class="chip"><b>Method</b>{html.escape(algorithm)}{f", k = {int(k)}" if algorithm != DBSCAN_ else f", eps = {eps:g}"}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if has_clusters:
    spend_col = next((c for c, r in profile.roles.items() if r == "monetary" and c.startswith("annual_spend")), None)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers analysed", f"{n_rows:,}")
    k2.metric("Segments found", metrics["clusters"])
    k3.metric("Silhouette score", "n/a" if np.isnan(metrics["silhouette"]) else f"{metrics['silhouette']:.3f}",
              help="Cluster separation, from -1 to 1. Above 0.25 is reasonable for behavioural data.")
    k4.metric("Unassigned customers", f"{metrics['noise_pct']:.1f}%" if algorithm == DBSCAN_ else "0%")
    k5.metric("Est. annual spend", format_inr(data[spend_col].sum()) if spend_col else "n/a",
              help="Orders per year x average order value, summed over all customers.")

col_labels = {c: display_name(c) for c in data.columns}

# ------------------------------------------------------------------------ tabs
tab_data, tab_select, tab_clusters, tab_insights, tab_compare, tab_export = st.tabs(
    ["Data", "Choose parameters", "Clusters", "Segments and actions", "Compare algorithms", "Export"])

# ---- Data -----------------------------------------------------------------
with tab_data:
    st.caption(f"Source: {dataset_name}")
    missing_series = raw_df[features + cat_features].isna().sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in file", f"{len(raw_df):,}")
    c2.metric("Missing cells", f"{missing_series.sum():,}", help="Missing values in the selected features before cleaning.")
    c3.metric("Outlier values clipped", f"{sum(pre.report['clipped_values'].values()):,}")
    c4.metric("Model features", n_dims, help="Numeric features plus encoded categorical columns.")

    left, right = st.columns(2)
    with left:
        if missing_series.sum() > 0:
            chart(viz.missing_values_figure(missing_series), "missing")
        else:
            st.success("No missing values in the selected features.")
    with right:
        chart(viz.correlation_figure(data[features]), "corr")

    st.subheader("What preprocessing did")
    prep_table = pd.DataFrame({
        "Feature": [display_name(c) for c in features],
        "Missing before": [pre.report["missing_before"][c] for c in features],
        "Values clipped": [pre.report["clipped_values"].get(c, 0) for c in features],
        "Log-transformed": ["yes" if c in pre.report["log_transformed"] else "no" for c in features],
    })
    st.dataframe(prep_table, hide_index=True)
    st.caption(f"Missing values: {MISSING_LABELS[missing_strategy].lower()}. Scaling: {SCALER_LABELS[scaler].lower()}"
               f"{', then L2 row normalisation' if l2 else ''}. Duplicates removed: {pre.report['duplicates_removed']}.")

    feat_pick = st.selectbox("Inspect a feature", features, format_func=display_name)
    ba_left, ba_right = st.columns([2, 1])
    with ba_left:
        chart(viz.before_after_figure(raw_df[feat_pick], pre.X_df[feat_pick], display_name(feat_pick)), "beforeafter")
    with ba_right:
        chart(viz.histogram_figure(data[feat_pick], display_name(feat_pick)), "hist")

    with st.expander("Preview of the cleaned data"):
        st.dataframe(data.head(100), hide_index=True)

# ---- Choose parameters ------------------------------------------------------
with tab_select:
    st.subheader("Elbow method and silhouette analysis")
    st.markdown(
        f'<div class="callout">The elbow of the inertia curve is at <b>k = {rec["elbow_k"]}</b> and the highest silhouette '
        f'score ({rec["best_silhouette"]:.3f}) is at <b>k = {rec["silhouette_k"]}</b>. The dashboard defaults to '
        f'<b>k = {rec["recommended_k"]}</b>; ties within 0.02 silhouette favour the elbow.</div>',
        unsafe_allow_html=True)
    scans = {"K-Means": kmeans_scan}
    if algorithm == HIER and n_rows <= MAX_HIERARCHICAL_ROWS:
        scans[f"Hierarchical ({linkage})"] = cached_scan(X, k_max, "hierarchical", linkage)
    e1, e2 = st.columns(2)
    with e1:
        chart(viz.elbow_figure(kmeans_scan, rec["elbow_k"]), "elbow")
    with e2:
        chart(viz.silhouette_scan_figure(scans, rec["silhouette_k"]), "silscan")
    scan_table = kmeans_scan.rename(columns={"k": "k", "inertia": "Inertia", "silhouette": "Silhouette",
                                             "davies_bouldin": "Davies-Bouldin (lower is better)",
                                             "calinski_harabasz": "Calinski-Harabasz (higher is better)"})
    st.dataframe(scan_table.round(3), hide_index=True)

    st.subheader("Hierarchical clustering: dendrogram")
    if n_rows <= MAX_HIERARCHICAL_ROWS:
        dendro_method = linkage if algorithm == HIER else "ward"
        chart(viz.dendrogram_figure(cached_linkage(X, dendro_method), int(k) if algorithm != DBSCAN_ else rec["recommended_k"]),
              "dendro")
        st.caption(f"Linkage: {dendro_method}. A long vertical gap before a merge marks a natural place to cut the tree.")
    else:
        st.info(f"Dendrograms are limited to {MAX_HIERARCHICAL_ROWS:,} customers.")

    st.subheader("DBSCAN: choosing eps and min_samples")
    ms_ref = default_min_samples(n_dims)
    d1, d2 = st.columns(2)
    with d1:
        curve = cached_kdistance(X, ms_ref)
        chart(viz.k_distance_figure(curve, suggest_eps(X, ms_ref), ms_ref), "kdist")
    with d2:
        grid = cached_grid(X, ms_ref)
        chart(viz.dbscan_grid_figure(grid), "grid")
    st.caption("Look for settings with several clusters and little noise. A silhouette that looks excellent with "
               "most customers marked as noise is not a useful segmentation.")

# ---- Clusters ---------------------------------------------------------------
with tab_clusters:
    if not has_clusters:
        st.warning("DBSCAN marked every customer as noise. Increase eps or lower min_samples in the sidebar.")
    else:
        d1, d2 = st.columns(2)
        d1.metric("Davies-Bouldin index", "n/a" if np.isnan(metrics["davies_bouldin"]) else f"{metrics['davies_bouldin']:.2f}",
                  help="Lower is better.")
        d2.metric("Calinski-Harabasz index", "n/a" if np.isnan(metrics["calinski_harabasz"]) else f"{metrics['calinski_harabasz']:,.0f}",
                  help="Higher is better.")
        if metrics["clusters"] < 2:
            st.warning("Only one segment was found, so quality metrics are not defined. Adjust the parameters.")

        view = st.radio("Projection", ["PCA (2 components)", "Two chosen features"], horizontal=True)
        hover = [c for c in ("customer_name", "city") if c in data.columns] + [f for f in features[:3]]
        if view.startswith("PCA"):
            fig = viz.scatter_figure(plot_df, "PC1", "PC2", hover=hover,
                                     x_label=f"PC1 ({pc_var[0]:.0%} of variance)",
                                     y_label=f"PC2 ({pc_var[1]:.0%} of variance)" if n_pc > 1 else "PC2",
                                     order=segment_order, title="Segments in principal-component space")
        else:
            fx, fy = st.columns(2)
            x_feat = fx.selectbox("X axis", features, index=0, format_func=display_name)
            y_feat = fy.selectbox("Y axis", features, index=min(1, len(features) - 1), format_func=display_name)
            fig = viz.scatter_figure(plot_df, x_feat, y_feat, hover=hover, x_label=display_name(x_feat),
                                     y_label=display_name(y_feat), order=segment_order, title="Segments by chosen features")
        chart(fig, "scatter2d")

        if n_pc >= 3:
            with st.expander("3-D view (first three principal components)"):
                chart(viz.scatter_figure(plot_df, "PC1", "PC2", z="PC3", hover=hover, order=segment_order,
                                         title="Segments in 3-D", x_label="PC1", y_label="PC2", z_label="PC3"), "scatter3d")

        s1, s2 = st.columns(2)
        with s1:
            counts = [int(profile.sizes[i.cluster_id]) for i in insights]
            chart(viz.segment_size_figure(segment_order, counts), "sizes")
        with s2:
            sil_values, sil_labels = silhouette_by_sample(X, labels)
            if len(sil_values):
                chart(viz.silhouette_samples_figure(sil_values, sil_labels, names, metrics["silhouette"]), "silsamples")
            else:
                st.info("Silhouette plot needs at least two segments.")

        dist_options = [c for c in profile.means.columns]
        dist_col = st.selectbox("Compare a metric across segments", dist_options,
                                index=dist_options.index(features[0]) if features[0] in dist_options else 0,
                                format_func=display_name)
        chart(viz.distribution_figure(plot_df, dist_col, display_name(dist_col), segment_order), "dist")

# ---- Segments and actions ---------------------------------------------------
with tab_insights:
    if not has_clusters:
        st.warning("No segments to describe yet.")
    else:
        if profile.spend_share_pct is not None:
            chart(viz.value_share_figure(segment_order, [float(profile.share_pct[i.cluster_id]) for i in insights],
                                         [float(profile.spend_share_pct[i.cluster_id]) for i in insights]), "valueshare")
        table = profile_table(profile, insights)
        cfg = {}
        for col in ("Share of customers (%)", "Share of est. spend (%)"):
            if col in table.columns:
                cfg[col] = st.column_config.ProgressColumn(col, min_value=0, max_value=100, format="%.1f")
        st.dataframe(table, hide_index=True, column_config=cfg)

        h1, h2 = st.columns(2)
        with h1:
            z_cols = [c for c in pre.report["features_used"] if c in profile.z.columns]
            chart(viz.profile_heatmap_figure(profile.z[z_cols], names, col_labels), "heat")
        with h2:
            chart(viz.radar_figure(profile.means[z_cols], names, col_labels), "radar")

        mix_options = categorical_candidates(data, id_col, max_levels=40)
        if mix_options:
            mix_col = st.selectbox("Segment mix by", mix_options, format_func=display_name)
            chart(viz.composition_figure(plot_df, mix_col, segment_order, display_name(mix_col)), "mix")

        st.subheader("Segment playbook")
        swatches = viz.color_map(segment_order)
        for ins in insights:
            meta = f"{ins.customers:,} customers · {ins.share_pct:.1f}% of base"
            if ins.spend_share_pct is not None:
                meta += f" · {ins.spend_share_pct:.1f}% of est. spend"
            st.markdown(
                f'<div class="seg-head"><span class="seg-swatch" style="background:{swatches[ins.name]}"></span>'
                f'<span class="seg-name">{html.escape(ins.name)}</span><span class="seg-meta">{meta}</span></div>',
                unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                st.markdown(ins.summary)
                st.markdown('<div class="seg-label">What sets them apart</div>', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {t}" for t in ins.traits))
            with b:
                st.markdown('<div class="seg-label">Recommended actions</div>', unsafe_allow_html=True)
                st.markdown("\n".join(f"- {t}" for t in ins.actions))
            st.markdown(f'<div class="kpi-line"><b>Metric to watch:</b> {html.escape(ins.kpi)}</div>', unsafe_allow_html=True)

# ---- Compare algorithms -----------------------------------------------------
with tab_compare:
    st.markdown("All three algorithms on the same preprocessed data, using the parameters in the sidebar "
                f"(k = {k}, linkage = {linkage}, eps = {eps:g}, min_samples = {min_samples}).")
    runs = {"K-Means": cached_cluster(X, KMEANS, int(k), linkage, float(eps), int(min_samples)),
            "DBSCAN": cached_cluster(X, DBSCAN_, int(k), linkage, float(eps), int(min_samples))}
    if n_rows <= MAX_HIERARCHICAL_ROWS:
        runs["Hierarchical"] = cached_cluster(X, HIER, int(k), linkage, float(eps), int(min_samples))
    rows = []
    for name, res in runs.items():
        m = evaluate_labels(X, res.labels)
        rows.append({"Algorithm": name, "Segments": m["clusters"], "Unassigned (%)": round(m["noise_pct"], 1),
                     "Silhouette": m["silhouette"], "Davies-Bouldin": m["davies_bouldin"],
                     "Calinski-Harabasz": m["calinski_harabasz"]})
    comp = pd.DataFrame(rows)
    st.dataframe(comp.round(3), hide_index=True)
    c_left, c_right = st.columns(2)
    with c_left:
        chart(viz.metric_comparison_figure(comp.dropna(subset=["Silhouette"])), "cmp_sil") if comp["Silhouette"].notna().any() else None
    with c_right:
        if len(runs) > 1:
            chart(viz.agreement_figure(agreement_matrix({n: r.labels for n, r in runs.items()})), "ari")
    st.caption("DBSCAN silhouette ignores unassigned customers, so it flatters the algorithm when many are left out; "
               "read it together with the unassigned share. An agreement close to 1 means two methods found the same groups.")

# ---- Export -----------------------------------------------------------------
with tab_export:
    if not has_clusters:
        st.warning("Nothing to export yet.")
    else:
        export_df = data.copy()
        export_df["segment_id"] = labels
        export_df["segment"] = plot_df["Segment"].values
        st.download_button("Download segmented customers (CSV)", export_df.to_csv(index=False).encode("utf-8"),
                           "segmented_customers.csv", "text/csv")
        st.download_button("Download segment profiles (CSV)", profile_table(profile, insights).to_csv(index=False).encode("utf-8"),
                           "segment_profiles.csv", "text/csv")
        report = insights_to_markdown(insights, context={
            "Dataset": dataset_name, "Customers": f"{n_rows:,}", "Algorithm": algorithm,
            "Silhouette": "n/a" if np.isnan(metrics["silhouette"]) else f"{metrics['silhouette']:.3f}"})
        st.download_button("Download business insights (Markdown)", report.encode("utf-8"),
                           "segment_insights.md", "text/markdown")
        st.caption("The segmented CSV contains the cleaned customer table plus segment_id and segment name.")
