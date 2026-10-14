import unittest

import numpy as np
import pandas as pd

from segmentation.clustering import run_dbscan, run_kmeans
from segmentation.data_loader import generate_synthetic_customers
from segmentation.insights import (NOISE_NAME, build_insights, classify_segment, insights_to_markdown,
                                   profile_segments, profile_table, segment_name_map)
from segmentation.preprocessing import PreprocessConfig, preprocess

FEATURES = ["age", "annual_income_lakh", "spending_score", "tenure_months", "orders_per_year",
            "avg_order_value_inr", "days_since_last_order"]


class InsightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = generate_synthetic_customers(800, seed=11)
        cls.pre = preprocess(df, PreprocessConfig(numeric_features=FEATURES, id_column="customer_id"))
        cls.labels = run_kmeans(cls.pre.X, 5).labels
        cls.profile = profile_segments(cls.pre.data, cls.labels, "customer_id")
        cls.insights = build_insights(cls.profile, FEATURES)

    def test_sizes_and_shares_add_up(self):
        self.assertEqual(int(self.profile.sizes.sum()), len(self.labels))
        self.assertAlmostEqual(float(self.profile.share_pct.sum()), 100.0)
        self.assertAlmostEqual(float(self.profile.spend_share_pct.sum()), 100.0)

    def test_expected_business_segments_are_found(self):
        names = {i.name for i in self.insights}
        for expected in ("Champions", "Dormant / at risk", "Affluent, low engagement"):
            self.assertIn(expected, names)

    def test_champions_hold_a_disproportionate_share_of_spend(self):
        champ = next(i for i in self.insights if i.name == "Champions")
        self.assertGreater(champ.spend_share_pct, champ.share_pct * 1.8)

    def test_every_segment_has_actions_traits_and_unique_name(self):
        self.assertEqual(len({i.name for i in self.insights}), len(self.insights))
        for ins in self.insights:
            self.assertEqual(len(ins.actions), 3)
            self.assertEqual(len(ins.traits), 3)
            self.assertTrue(ins.kpi)

    def test_classification_grid(self):
        self.assertEqual(classify_segment(1.0, 1.0), "Champions")
        self.assertEqual(classify_segment(1.0, -1.0), "Affluent, low engagement")
        self.assertEqual(classify_segment(-1.0, 1.0), "Enthusiastic budget shoppers")
        self.assertEqual(classify_segment(-1.0, -1.0), "Dormant / at risk")
        self.assertEqual(classify_segment(0.0, 0.0), "Core mainstream")
        self.assertEqual(classify_segment(None, None), "Segment")

    def test_noise_segment_and_unknown_columns_fall_back_gracefully(self):
        labels = self.labels.copy(); labels[:30] = -1
        ins = build_insights(profile_segments(self.pre.data, labels, "customer_id"), FEATURES)
        self.assertEqual(ins[-1].name, NOISE_NAME)
        generic = pd.DataFrame({"foo": np.random.default_rng(0).normal(size=200),
                                "bar": np.random.default_rng(1).normal(size=200)})
        gl = (generic["foo"] > 0).astype(int).values
        g_ins = build_insights(profile_segments(generic, gl), ["foo", "bar"])
        self.assertTrue(all(i.name.startswith("Segment") for i in g_ins))

    def test_tables_and_report(self):
        table = profile_table(self.profile, self.insights)
        self.assertEqual(len(table), len(self.insights))
        self.assertIn("Share of est. spend (%)", table.columns)
        md = insights_to_markdown(self.insights, context={"Algorithm": "K-Means"})
        self.assertIn("## Champions", md)
        self.assertEqual(set(segment_name_map(self.insights)), set(range(5)))

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            profile_segments(self.pre.data, self.labels[:-1])


try:
    import plotly  # noqa: F401
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False


@unittest.skipUnless(HAVE_PLOTLY, "plotly not installed")
class VisualizationTests(unittest.TestCase):
    def test_every_figure_builds(self):
        from segmentation import visualization as viz
        from segmentation.clustering import (compute_linkage_matrix, dbscan_grid_search, default_dbscan_grid,
                                             k_distance_curve)
        from segmentation.evaluation import (agreement_matrix, cluster_scan, evaluate_labels, recommend_k,
                                             silhouette_by_sample)
        df = generate_synthetic_customers(400, seed=2)
        pre = preprocess(df, PreprocessConfig(numeric_features=FEATURES, id_column="customer_id"))
        X, data = pre.X, pre.data
        res = run_kmeans(X, 4)
        prof = profile_segments(data, res.labels, "customer_id")
        ins = build_insights(prof, FEATURES)
        names = segment_name_map(ins)
        order = [names[i.cluster_id] for i in ins]
        scan = cluster_scan(X, range(2, 7))
        plot_df = data.copy(); plot_df["Segment"] = [names[int(c)] for c in res.labels]
        pcs, var = viz.pca_projection(X, 3)
        for i in range(3):
            plot_df[f"PC{i + 1}"] = pcs[:, i]
        sil, lab = silhouette_by_sample(X, res.labels)
        cols = list(prof.z.columns)
        grid = dbscan_grid_search(X, *default_dbscan_grid(X, 14))
        figs = [
            viz.missing_values_figure(pre.features_raw.isna().sum() + 1),
            viz.correlation_figure(data[FEATURES]), viz.histogram_figure(data["age"], "Age"),
            viz.before_after_figure(data["age"], pre.X_df["age"], "Age"),
            viz.elbow_figure(scan, 4), viz.silhouette_scan_figure({"K-Means": scan}, 4),
            viz.k_distance_figure(k_distance_curve(X, 14), 1.0, 14), viz.dbscan_grid_figure(grid),
            viz.dendrogram_figure(compute_linkage_matrix(X), 4),
            viz.scatter_figure(plot_df, "PC1", "PC2", order=order),
            viz.scatter_figure(plot_df, "PC1", "PC2", z="PC3", order=order),
            viz.segment_size_figure(order, [int(prof.sizes[i.cluster_id]) for i in ins]),
            viz.silhouette_samples_figure(sil, lab, names, evaluate_labels(X, res.labels)["silhouette"]),
            viz.distribution_figure(plot_df, "age", "Age", order),
            viz.profile_heatmap_figure(prof.z, names, {}), viz.radar_figure(prof.means[FEATURES], names, {}),
            viz.agreement_figure(agreement_matrix({"A": res.labels, "B": res.labels})),
            viz.metric_comparison_figure(pd.DataFrame({"Algorithm": ["A", "B"], "Silhouette": [0.3, 0.4]})),
        ]
        for fig in figs:
            self.assertGreater(len(fig.data), 0)
        self.assertEqual(viz.color_map(["a", NOISE_NAME])[NOISE_NAME], viz.NOISE_COLOR)


try:
    import streamlit  # noqa: F401
    HAVE_STREAMLIT = True
except ImportError:
    HAVE_STREAMLIT = False


@unittest.skipUnless(HAVE_STREAMLIT and HAVE_PLOTLY, "streamlit/plotly not installed")
class DashboardSmokeTests(unittest.TestCase):
    """Runs app.py headlessly with Streamlit's AppTest and fails on any exception."""

    def test_app_runs_for_each_algorithm(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=180).run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        for method in ("Hierarchical", "DBSCAN"):
            at.sidebar.selectbox[[s.label for s in at.sidebar.selectbox].index("Method")].select(method).run()
            self.assertFalse(at.exception, [e.value for e in at.exception])


if __name__ == "__main__":
    unittest.main()
