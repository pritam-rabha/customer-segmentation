"""K-Means, DBSCAN and Hierarchical clustering plus parameter-selection helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors

from .config import MAX_HIERARCHICAL_ROWS, RANDOM_STATE
from .utils import find_knee

NOISE = -1
LINKAGES = ("ward", "complete", "average", "single")


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(len(set(self.labels.tolist()) - {NOISE}))

    @property
    def n_noise(self) -> int:
        return int((self.labels == NOISE).sum())


def relabel_by_size(labels) -> np.ndarray:
    """Renumber clusters 0..k-1 by descending size; noise stays -1."""
    labels = np.asarray(labels)
    out = np.full(labels.shape, NOISE, dtype=int)
    valid = labels[labels != NOISE]
    if valid.size == 0:
        return out
    ids, counts = np.unique(valid, return_counts=True)
    for new_id, old_id in enumerate(ids[np.argsort(-counts, kind="stable")]):
        out[labels == old_id] = new_id
    return out


def _check_k(X, k: int) -> None:
    if not 2 <= k < len(X):
        raise ValueError(f"n_clusters must be between 2 and {len(X) - 1}; got {k}.")


def run_kmeans(X, n_clusters: int, random_state: int = RANDOM_STATE, n_init: int = 10) -> ClusterResult:
    _check_k(X, n_clusters)
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state).fit(X)
    return ClusterResult("K-Means", relabel_by_size(model.labels_),
                         {"n_clusters": n_clusters, "inertia": float(model.inertia_)})


def run_hierarchical(X, n_clusters: int, linkage: str = "ward") -> ClusterResult:
    _check_k(X, n_clusters)
    if linkage not in LINKAGES:
        raise ValueError(f"linkage must be one of {LINKAGES}")
    if len(X) > MAX_HIERARCHICAL_ROWS:
        raise ValueError(
            f"Hierarchical clustering is limited to {MAX_HIERARCHICAL_ROWS:,} rows "
            f"(got {len(X):,}). Use K-Means or sample the data."
        )
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)
    return ClusterResult("Hierarchical", relabel_by_size(model.labels_),
                         {"n_clusters": n_clusters, "linkage": linkage})


def run_dbscan(X, eps: float, min_samples: int) -> ClusterResult:
    if eps <= 0 or min_samples < 1:
        raise ValueError("eps must be > 0 and min_samples >= 1.")
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return ClusterResult("DBSCAN", relabel_by_size(model.labels_),
                         {"eps": float(eps), "min_samples": int(min_samples)})


def compute_linkage_matrix(X, method: str = "ward") -> np.ndarray:
    """SciPy linkage matrix used for the dendrogram."""
    if len(X) > MAX_HIERARCHICAL_ROWS:
        raise ValueError(f"Dendrogram is limited to {MAX_HIERARCHICAL_ROWS:,} rows.")
    return scipy_linkage(X, method=method)


# ------------------------------------------------------------- DBSCAN helpers
def default_min_samples(n_features: int) -> int:
    """Common rule of thumb: at least 2 x dimensions, never below 4."""
    return max(4, 2 * n_features)


def k_distance_curve(X, k: int) -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour."""
    k = min(k, len(X) - 1)
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(distances[:, -1])


def suggest_eps(X, min_samples: int, fallback: float = 0.5) -> float:
    """Pick eps at the knee of the k-distance curve.

    Duplicate customers (common with one-hot or low-cardinality features) make the start of
    the curve exactly zero; only the positive part is searched, and the result is always > 0.
    """
    curve = k_distance_curve(X, min_samples)
    positive = curve[curve > 0]
    if positive.size < 3:
        return float(fallback)
    knee = positive[find_knee(np.arange(len(positive)), positive)]
    return float(round(max(knee, 1e-3), 3))


def dbscan_grid_search(X, eps_values, min_samples_values) -> pd.DataFrame:
    """Try a grid of DBSCAN settings and report the resulting structure."""
    from .evaluation import safe_silhouette  # local import avoids a circular dependency

    rows = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            if eps <= 0:
                continue
            labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(X)
            n_clusters = len(set(labels.tolist()) - {NOISE})
            rows.append({
                "eps": float(eps),
                "min_samples": int(min_samples),
                "clusters": n_clusters,
                "noise_pct": float((labels == NOISE).mean() * 100),
                "silhouette": safe_silhouette(X, labels),
            })
    return pd.DataFrame(rows)


def default_dbscan_grid(X, min_samples: int) -> tuple[np.ndarray, list[int]]:
    eps0 = max(suggest_eps(X, min_samples), 0.01)
    eps_values = np.unique(np.round(np.linspace(eps0 * 0.5, eps0 * 1.8, 8), 3))
    eps_values = eps_values[eps_values > 0]
    ms_values = sorted({max(3, min_samples - 3), min_samples, min_samples + 4, min_samples + 8})
    return eps_values, ms_values


def suggest_dbscan_params(grid: pd.DataFrame, max_noise_pct: float = 20.0) -> tuple[float, int] | None:
    """Best (eps, min_samples) from a grid: >=2 clusters, limited noise, top silhouette.

    Returns None when no setting in the grid yields at least two clusters.
    """
    ok = grid[(grid["clusters"] >= 2) & (grid["noise_pct"] <= max_noise_pct) & grid["silhouette"].notna()]
    if ok.empty:
        ok = grid[(grid["clusters"] >= 2) & grid["silhouette"].notna()]
        if ok.empty:
            return None
        ok = ok.sort_values("noise_pct").head(1)
    best = ok.sort_values(["silhouette", "clusters"], ascending=False).iloc[0]
    return float(best["eps"]), int(best["min_samples"])
