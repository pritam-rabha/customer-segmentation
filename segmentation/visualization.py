"""Plotly figures for the dashboard. Every function returns a ``go.Figure``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster import hierarchy as sch
from sklearn.decomposition import PCA

from .insights import NOISE_NAME
from .schema import display_name

PALETTE = ["#2F4B9A", "#D08C2F", "#3A7D5C", "#8E4B7A", "#5E7184", "#B23A48", "#3E9AA0", "#8A7A35", "#6B5CA5", "#9A6B4F"]
NOISE_COLOR = "#B8BEC6"
INK, GRID = "#1B2430", "#E3E7ED"
FONT = "Public Sans, Segoe UI, Helvetica, Arial, sans-serif"
DIVERGING = [[0.0, "#2F4B9A"], [0.5, "#F5F6F8"], [1.0, "#C07A1C"]]


def color_map(names) -> dict[str, str]:
    """Stable colour per segment name; noise is always light grey."""
    cmap, i = {}, 0
    for name in names:
        if name == NOISE_NAME:
            cmap[name] = NOISE_COLOR
        else:
            cmap[name] = PALETTE[i % len(PALETTE)]
            i += 1
    return cmap


def _style(fig: go.Figure, title: str | None = None, height: int = 380, legend: bool = True) -> go.Figure:
    layout = dict(
        template="plotly_white",
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=52, r=16, t=56 if title else 28, b=48),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(size=15))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ------------------------------------------------------------------ data overview
def missing_values_figure(missing: pd.Series) -> go.Figure:
    s = missing[missing > 0].sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=PALETTE[0],
                           hovertemplate="%{y}: %{x} missing<extra></extra>"))
    fig.update_xaxes(title_text="Missing values")
    return _style(fig, "Missing values by column", height=max(220, 40 * len(s) + 90), legend=False)


def correlation_figure(df: pd.DataFrame) -> go.Figure:
    corr = df.corr(numeric_only=True)
    corr.index = corr.columns = [display_name(c) for c in corr.columns]
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=DIVERGING, text_auto=".2f", aspect="auto")
    fig.update_coloraxes(colorbar_title_text="r")
    return _style(fig, "Feature correlation", height=420, legend=False)


def histogram_figure(series: pd.Series, label: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series, marker_color=PALETTE[0], nbinsx=30))
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Customers")
    return _style(fig, f"Distribution of {label.lower()}", height=300, legend=False)


def before_after_figure(before: pd.Series, after: pd.Series, label: str) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Before scaling", "After scaling"))
    fig.add_trace(go.Histogram(x=before, marker_color=PALETTE[4], nbinsx=30), row=1, col=1)
    fig.add_trace(go.Histogram(x=after, marker_color=PALETTE[0], nbinsx=30), row=1, col=2)
    fig.update_yaxes(title_text="Customers", row=1, col=1)
    return _style(fig, f"{label}: effect of preprocessing", height=300, legend=False)


# ------------------------------------------------------------- choosing parameters
def elbow_figure(scan: pd.DataFrame, elbow_k: int | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=scan["k"], y=scan["inertia"], mode="lines+markers",
                               line=dict(color=PALETTE[0], width=2), marker=dict(size=7),
                               hovertemplate="k=%{x}<br>Inertia=%{y:,.0f}<extra></extra>"))
    if elbow_k is not None:
        y = float(scan.loc[scan["k"] == elbow_k, "inertia"].iloc[0])
        fig.add_trace(go.Scatter(x=[elbow_k], y=[y], mode="markers", name=f"Elbow (k={elbow_k})",
                                 marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=PALETTE[1], width=3))))
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title_text="Within-cluster sum of squares")
    return _style(fig, "Elbow method (K-Means)", legend=elbow_k is not None)


def silhouette_scan_figure(scans: dict[str, pd.DataFrame], best_k: int | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (name, scan) in enumerate(scans.items()):
        fig.add_trace(go.Scatter(x=scan["k"], y=scan["silhouette"], mode="lines+markers", name=name,
                                 line=dict(color=PALETTE[i], width=2), marker=dict(size=7),
                                 hovertemplate="k=%{x}<br>Silhouette=%{y:.3f}<extra></extra>"))
    if best_k is not None:
        fig.add_vline(x=best_k, line_dash="dot", line_color=PALETTE[1],
                      annotation_text=f"best k = {best_k}", annotation_position="top")
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title_text="Mean silhouette score")
    return _style(fig, "Silhouette analysis")


def k_distance_figure(curve: np.ndarray, eps: float, min_samples: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=np.arange(len(curve)), y=curve, mode="lines", line=dict(color=PALETTE[0], width=2)))
    fig.add_hline(y=eps, line_dash="dot", line_color=PALETTE[1], annotation_text=f"eps = {eps:g}")
    fig.update_xaxes(title_text="Customers, sorted by distance")
    fig.update_yaxes(title_text=f"Distance to neighbour #{min_samples}")
    return _style(fig, "k-distance plot (DBSCAN eps)", legend=False)


def dbscan_grid_figure(grid: pd.DataFrame) -> go.Figure:
    sil = grid.pivot(index="min_samples", columns="eps", values="silhouette")
    clusters = grid.pivot(index="min_samples", columns="eps", values="clusters")
    noise = grid.pivot(index="min_samples", columns="eps", values="noise_pct")
    labels = clusters.astype(int).astype(str) + " cl / " + noise.round(0).astype(int).astype(str) + "% noise"
    fig = go.Figure(go.Heatmap(
        z=sil.values, x=[f"{c:g}" for c in sil.columns], y=[str(i) for i in sil.index],
        text=labels.values, texttemplate="%{text}", colorscale=[[0.0, "#F5F6F8"], [1.0, PALETTE[0]]], colorbar=dict(title=dict(text="Silhouette")),
        hovertemplate="eps=%{x}<br>min_samples=%{y}<br>silhouette=%{z:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="eps", type="category")
    fig.update_yaxes(title_text="min_samples", type="category")
    return _style(fig, "DBSCAN parameter search (cells show clusters / noise)", height=360, legend=False)


def dendrogram_figure(Z: np.ndarray, n_clusters: int | None = None, p: int = 40) -> go.Figure:
    d = sch.dendrogram(Z, truncate_mode="lastp", p=p, no_plot=True)
    fig = go.Figure()
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=PALETTE[0], width=1.2),
                                 hoverinfo="skip", showlegend=False))
    if n_clusters and 2 <= n_clusters <= len(Z):
        cut = (Z[-(n_clusters - 1), 2] + Z[-n_clusters, 2]) / 2 if n_clusters > 2 else (Z[-1, 2] + Z[-2, 2]) / 2
        fig.add_hline(y=cut, line_dash="dot", line_color=PALETTE[1], annotation_text=f"cut for k = {n_clusters}")
    fig.update_xaxes(showticklabels=False, title_text=f"Customers (last {p} merged branches)", showgrid=False)
    fig.update_yaxes(title_text="Merge distance")
    return _style(fig, "Dendrogram (truncated)", legend=False)


# ----------------------------------------------------------------------- clusters
def pca_projection(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def scatter_figure(plot_df: pd.DataFrame, x: str, y: str, z: str | None = None, hover: list[str] | None = None,
                   x_label: str | None = None, y_label: str | None = None, z_label: str | None = None,
                   title: str = "Customer segments", order: list[str] | None = None) -> go.Figure:
    """2-D (or 3-D when ``z`` is given) scatter coloured by the ``Segment`` column."""
    order = order or list(dict.fromkeys(plot_df["Segment"]))
    kwargs = dict(color="Segment", color_discrete_map=color_map(order), category_orders={"Segment": order},
                  hover_data=hover or [], labels={x: x_label or x, y: y_label or y})
    if z:
        kwargs["labels"][z] = z_label or z
        fig = px.scatter_3d(plot_df, x=x, y=y, z=z, **kwargs)
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        fig.update_layout(scene=dict(xaxis=dict(title=dict(text=x_label or x)), yaxis=dict(title=dict(text=y_label or y)),
                                     zaxis=dict(title=dict(text=z_label or z))))
        return _style(fig, title, height=560)
    fig = px.scatter(plot_df, x=x, y=y, **kwargs)
    fig.update_traces(marker=dict(size=7, opacity=0.78, line=dict(width=0)))
    return _style(fig, title, height=500)


def segment_size_figure(names: list[str], counts: list[int]) -> go.Figure:
    cmap = color_map(names)
    total = sum(counts)
    fig = go.Figure(go.Bar(
        x=counts, y=names, orientation="h", marker_color=[cmap[n] for n in names],
        text=[f"{c:,} ({c / total:.0%})" for c in counts], textposition="auto",
        hovertemplate="%{y}: %{x:,} customers<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Customers")
    return _style(fig, "Segment sizes", height=max(240, 52 * len(names) + 90), legend=False)


def silhouette_samples_figure(values: np.ndarray, labels: np.ndarray, names: dict[int, str], average: float) -> go.Figure:
    order = [names[c] for c in sorted(names) if c in set(labels.tolist())]
    cmap = color_map(order)
    fig, offset = go.Figure(), 0
    for cid in sorted(set(labels.tolist())):
        vals = np.sort(values[labels == cid])
        y = np.arange(offset, offset + len(vals))
        fig.add_trace(go.Scatter(x=vals, y=y, mode="lines", fill="tozerox", name=names[cid],
                                 line=dict(color=cmap[names[cid]], width=1)))
        offset += len(vals) + max(10, len(values) // 50)
    fig.add_vline(x=average, line_dash="dot", line_color=INK, annotation_text=f"mean {average:.2f}")
    fig.update_xaxes(title_text="Silhouette coefficient")
    fig.update_yaxes(showticklabels=False, title_text="Customers by segment")
    return _style(fig, "Silhouette by customer", height=420)


def distribution_figure(plot_df: pd.DataFrame, column: str, label: str, order: list[str]) -> go.Figure:
    fig = px.box(plot_df, x="Segment", y=column, color="Segment", color_discrete_map=color_map(order),
                 category_orders={"Segment": order}, points=False, labels={column: label})
    fig.update_xaxes(title_text="")
    return _style(fig, f"{label} by segment", height=380, legend=False)


# ------------------------------------------------------------------------ profiles
def profile_heatmap_figure(z: pd.DataFrame, names: dict[int, str], col_labels: dict[str, str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[col_labels.get(c, c) for c in z.columns], y=[names[i] for i in z.index],
        zmin=-2, zmax=2, colorscale=DIVERGING, text=np.round(z.values, 1), texttemplate="%{text}",
        colorbar=dict(title=dict(text="Std. dev. from average")),
        hovertemplate="%{y}<br>%{x}: %{z:+.2f} SD<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "How each segment differs from the average customer", height=max(280, 60 * len(z) + 120), legend=False)


def radar_figure(means: pd.DataFrame, names: dict[int, str], col_labels: dict[str, str]) -> go.Figure:
    span = (means.max() - means.min()).replace(0, np.nan)
    scaled = ((means - means.min()) / span).fillna(0.5)
    theta = [col_labels.get(c, c) for c in scaled.columns]
    order = [names[i] for i in scaled.index]
    cmap = color_map(order)
    fig = go.Figure()
    for cid in scaled.index:
        vals = scaled.loc[cid].tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=theta + theta[:1], name=names[cid],
                                      line=dict(color=cmap[names[cid]], width=2), fill="toself",
                                      opacity=0.35 if len(scaled) > 3 else 0.45))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=GRID),
                                 angularaxis=dict(gridcolor=GRID)))
    return _style(fig, "Segment profiles (each axis scaled min to max across segments)", height=520)


def agreement_figure(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix.values, x=list(matrix.columns), y=list(matrix.index), zmin=0, zmax=1,
                               colorscale=[[0, "#F5F6F8"], [1, PALETTE[0]]], text=np.round(matrix.values, 2),
                               texttemplate="%{text}", colorbar=dict(title=dict(text="ARI"))))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Agreement between algorithms (Adjusted Rand Index)", height=340, legend=False)


def metric_comparison_figure(table: pd.DataFrame) -> go.Figure:
    """Grouped bars of silhouette per algorithm."""
    fig = go.Figure(go.Bar(x=table["Algorithm"], y=table["Silhouette"], marker_color=PALETTE[: len(table)],
                           text=table["Silhouette"].round(3), textposition="outside"))
    fig.update_yaxes(title_text="Mean silhouette (noise excluded)", rangemode="tozero")
    return _style(fig, "Silhouette by algorithm", height=340, legend=False)


CATEGORY_RAMP = ["#2F4B9A", "#4F67AD", "#6F84C0", "#92A2D2", "#B5C0E3", "#D3D9EE", "#9AA3AE", "#C0C6CE", "#E2E5E9"]


def value_share_figure(names: list[str], customer_share: list[float], spend_share: list[float]) -> go.Figure:
    """Share of customers vs share of estimated spend per segment (revenue concentration)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Share of customers", x=names, y=customer_share, marker_color="#B5C0E3",
                         text=[f"{v:.0f}%" for v in customer_share], textposition="outside"))
    fig.add_trace(go.Bar(name="Share of est. annual spend", x=names, y=spend_share, marker_color=PALETTE[0],
                         text=[f"{v:.0f}%" for v in spend_share], textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(title_text="% of total", rangemode="tozero")
    return _style(fig, "Who generates the revenue", height=380)


def composition_figure(plot_df: pd.DataFrame, column: str, order: list[str], label: str, top: int = 8) -> go.Figure:
    """100% stacked bars: category mix inside every segment (top categories, rest grouped)."""
    values = plot_df[column].astype(str)
    keep = values.value_counts().index[:top]
    values = values.where(values.isin(keep), "Other")
    ct = pd.crosstab(plot_df["Segment"], values, normalize="index").reindex(order) * 100
    fig = go.Figure()
    for i, cat in enumerate(ct.columns):
        fig.add_trace(go.Bar(name=str(cat), y=ct.index, x=ct[cat], orientation="h",
                             marker_color=CATEGORY_RAMP[i % len(CATEGORY_RAMP)],
                             hovertemplate="%{y}<br>" + str(cat) + ": %{x:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="% of segment", range=[0, 100])
    return _style(fig, f"{label} mix within each segment", height=max(300, 56 * len(order) + 130))
