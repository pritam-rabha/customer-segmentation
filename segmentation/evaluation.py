"""Cluster-quality analysis: Elbow method, Silhouette score and companions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples, silhouette_score)

from .clustering import NOISE, run_hierarchical, run_kmeans
from .config import RANDOM_STATE, SILHOUETTE_SAMPLE_SIZE
from .utils import find_knee


def _non_noise(X, labels):
    labels = np.asarray(labels)
    mask = labels != NOISE
    return np.asarray(X)[mask], labels[mask]


def _valid(labels) -> bool:
    return len(np.unique(labels)) >= 2


def safe_silhouette(X, labels) -> float:
    """Mean silhouette ignoring DBSCAN noise; NaN when fewer than two clusters exist."""
    Xv, lv = _non_noise(X, labels)
    if not _valid(lv) or len(Xv) <= len(np.unique(lv)):
        return float("nan")
    sample = SILHOUETTE_SAMPLE_SIZE if len(Xv) > SILHOUETTE_SAMPLE_SIZE else None
    return float(silhouette_score(Xv, lv, sample_size=sample, random_state=RANDOM_STATE))


def evaluate_labels(X, labels) -> dict:
    """Internal validity metrics for one clustering (noise excluded)."""
    labels = np.asarray(labels)
    out = {
        "clusters": int(len(set(labels.tolist()) - {NOISE})),
        "noise_points": int((labels == NOISE).sum()),
        "noise_pct": float((labels == NOISE).mean() * 100),
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    Xv, lv = _non_noise(X, labels)
    if _valid(lv) and len(Xv) > len(np.unique(lv)):
        out["silhouette"] = safe_silhouette(X, labels)
        out["davies_bouldin"] = float(davies_bouldin_score(Xv, lv))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xv, lv))
    return out


def silhouette_by_sample(X, labels) -> tuple[np.ndarray, np.ndarray]:
    """Per-point silhouette values (noise excluded) and the matching labels."""
    Xv, lv = _non_noise(X, labels)
    if not _valid(lv):
        return np.array([]), np.array([])
    return silhouette_samples(Xv, lv), lv


def cluster_scan(X, k_range=range(2, 11), algorithm: str = "kmeans", linkage: str = "ward") -> pd.DataFrame:
    """Evaluate every k in ``k_range``: inertia (K-Means), silhouette, DB and CH."""
    rows = []
    for k in k_range:
        if algorithm == "kmeans":
            res = run_kmeans(X, k)
        elif algorithm == "hierarchical":
            res = run_hierarchical(X, k, linkage)
        else:
            raise ValueError("algorithm must be 'kmeans' or 'hierarchical'")
        metrics = evaluate_labels(X, res.labels)
        rows.append({
            "k": k,
            "inertia": res.params.get("inertia", float("nan")),
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "calinski_harabasz": metrics["calinski_harabasz"],
        })
    return pd.DataFrame(rows)


def find_elbow_k(scan: pd.DataFrame) -> int:
    """k at the elbow of the K-Means inertia curve."""
    idx = find_knee(scan["k"].values, scan["inertia"].values)
    return int(scan["k"].iloc[idx])


def recommend_k(scan: pd.DataFrame, tolerance: float = 0.02) -> dict:
    """Combine both methods.

    The silhouette-optimal k is chosen; when the elbow k scores within ``tolerance``
    of it the (usually simpler) elbow k wins the tie.
    """
    best_idx = int(scan["silhouette"].idxmax())
    best_k = int(scan.loc[best_idx, "k"])
    best_sil = float(scan.loc[best_idx, "silhouette"])
    elbow_k = find_elbow_k(scan) if scan["inertia"].notna().all() else best_k
    elbow_sil = float(scan.loc[scan["k"] == elbow_k, "silhouette"].iloc[0])
    chosen = elbow_k if elbow_sil >= best_sil - tolerance else best_k
    return {"silhouette_k": best_k, "elbow_k": elbow_k, "recommended_k": chosen,
            "best_silhouette": best_sil}


def agreement_matrix(label_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    """Adjusted Rand Index between every pair of clusterings (1 = identical)."""
    names = list(label_sets)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            mat.loc[a, b] = mat.loc[b, a] = adjusted_rand_score(label_sets[a], label_sets[b])
    return mat
