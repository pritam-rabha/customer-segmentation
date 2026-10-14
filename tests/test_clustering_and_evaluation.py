import unittest

import numpy as np
from sklearn.metrics import adjusted_rand_score

from segmentation.clustering import (NOISE, compute_linkage_matrix, dbscan_grid_search, default_dbscan_grid,
                                     default_min_samples, k_distance_curve, relabel_by_size, run_dbscan,
                                     run_hierarchical, run_kmeans, suggest_dbscan_params, suggest_eps)
from segmentation.data_loader import generate_synthetic_customers
from segmentation.evaluation import (agreement_matrix, cluster_scan, evaluate_labels, find_elbow_k,
                                     recommend_k, safe_silhouette, silhouette_by_sample)
from segmentation.preprocessing import PreprocessConfig, preprocess
from segmentation.utils import find_knee

FEATURES = ["age", "annual_income_lakh", "spending_score", "tenure_months", "orders_per_year",
            "avg_order_value_inr", "days_since_last_order"]


class ClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df, cls.truth = generate_synthetic_customers(600, seed=7, return_truth=True)
        cls.X = preprocess(df, PreprocessConfig(numeric_features=FEATURES, id_column="customer_id")).X

    def test_kmeans_recovers_known_groups(self):
        res = run_kmeans(self.X, 5)
        self.assertEqual(res.n_clusters, 5)
        self.assertGreater(adjusted_rand_score(self.truth, res.labels), 0.8)

    def test_hierarchical_recovers_known_groups(self):
        res = run_hierarchical(self.X, 5, "ward")
        self.assertEqual(res.n_clusters, 5)
        self.assertGreater(adjusted_rand_score(self.truth, res.labels), 0.8)

    def test_dbscan_finds_structure_and_noise_label(self):
        ms = default_min_samples(self.X.shape[1])
        grid = dbscan_grid_search(self.X, *default_dbscan_grid(self.X, ms))
        eps, ms = suggest_dbscan_params(grid)
        res = run_dbscan(self.X, eps, ms)
        self.assertGreaterEqual(res.n_clusters, 2)
        self.assertEqual(res.n_noise, int((res.labels == NOISE).sum()))

    def test_dbscan_on_tiny_eps_is_all_noise(self):
        res = run_dbscan(self.X, 0.01, 5)
        self.assertEqual(res.n_clusters, 0)
        self.assertEqual(res.n_noise, len(self.X))

    def test_labels_are_ordered_by_size(self):
        sizes = np.bincount(run_kmeans(self.X, 4).labels)
        self.assertTrue(all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1)))

    def test_relabel_keeps_noise(self):
        out = relabel_by_size(np.array([5, 5, 5, 2, 2, -1, 9]))
        self.assertEqual(out.tolist(), [0, 0, 0, 1, 1, -1, 2])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run_kmeans(self.X, 1)
        with self.assertRaises(ValueError):
            run_hierarchical(self.X, 3, "manhattan")
        with self.assertRaises(ValueError):
            run_dbscan(self.X, -1, 5)

    def test_k_distance_and_linkage_shapes(self):
        self.assertEqual(len(k_distance_curve(self.X, 5)), len(self.X))
        self.assertGreater(suggest_eps(self.X, 5), 0)
        self.assertEqual(compute_linkage_matrix(self.X).shape, (len(self.X) - 1, 4))


class DuplicateHeavyDataTests(unittest.TestCase):
    """Categorical-only features produce many identical customers (k-distance of exactly 0)."""

    def test_eps_and_grid_stay_positive(self):
        df = generate_synthetic_customers(500, seed=4)
        X = preprocess(df, PreprocessConfig(numeric_features=[], id_column="customer_id",
                                            categorical_features=["gender", "city_tier", "preferred_payment"])).X
        ms = default_min_samples(X.shape[1])
        self.assertEqual(int((k_distance_curve(X, ms) == 0).sum()) > 0, True)
        self.assertGreater(suggest_eps(X, ms), 0)
        eps_values, ms_values = default_dbscan_grid(X, ms)
        self.assertTrue((eps_values > 0).all())
        grid = dbscan_grid_search(X, eps_values, ms_values)   # must not raise
        self.assertEqual(len(grid), len(eps_values) * len(ms_values))
        self.assertTrue((grid["eps"] > 0).all())

    def test_grid_search_skips_nonpositive_eps(self):
        X = np.random.default_rng(0).normal(size=(60, 2))
        grid = dbscan_grid_search(X, [0.0, 0.5], [4])
        self.assertEqual(grid["eps"].tolist(), [0.5])


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = generate_synthetic_customers(600, seed=7)
        cls.X = preprocess(df, PreprocessConfig(numeric_features=FEATURES, id_column="customer_id")).X

    def test_scan_selects_five_clusters_on_five_group_data(self):
        scan = cluster_scan(self.X, range(2, 9))
        rec = recommend_k(scan)
        self.assertEqual(rec["silhouette_k"], 5)
        self.assertIn(rec["recommended_k"], (4, 5))
        self.assertTrue((scan["inertia"].diff().dropna() < 0).all())  # inertia must fall as k grows

    def test_hierarchical_scan_has_no_inertia(self):
        scan = cluster_scan(self.X, range(2, 5), "hierarchical")
        self.assertTrue(scan["inertia"].isna().all())
        self.assertTrue(scan["silhouette"].between(-1, 1).all())

    def test_elbow_on_synthetic_curve(self):
        import pandas as pd
        scan = pd.DataFrame({"k": range(2, 9), "inertia": [100, 60, 30, 25, 22, 20, 19]})
        self.assertEqual(find_elbow_k(scan), 4)
        self.assertEqual(find_knee([0, 1, 2], [0, 0.1, 1]), 1)

    def test_metrics_ignore_noise_and_handle_single_cluster(self):
        labels = np.array(run_kmeans(self.X, 3).labels)
        labels[:20] = NOISE
        m = evaluate_labels(self.X, labels)
        self.assertEqual(m["noise_points"], 20)
        self.assertTrue(-1 <= m["silhouette"] <= 1)
        single = evaluate_labels(self.X, np.zeros(len(self.X), dtype=int))
        self.assertTrue(np.isnan(single["silhouette"]))
        self.assertTrue(np.isnan(safe_silhouette(self.X, np.full(len(self.X), NOISE))))

    def test_silhouette_per_sample_and_agreement(self):
        a, b = run_kmeans(self.X, 5).labels, run_hierarchical(self.X, 5).labels
        values, lab = silhouette_by_sample(self.X, a)
        self.assertEqual(len(values), len(lab))
        mat = agreement_matrix({"A": a, "B": b})
        self.assertEqual(mat.loc["A", "A"], 1.0)
        self.assertAlmostEqual(mat.loc["A", "B"], mat.loc["B", "A"])


if __name__ == "__main__":
    unittest.main()
