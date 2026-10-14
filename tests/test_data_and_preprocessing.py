import io
import unittest

import numpy as np
import pandas as pd

from segmentation.data_loader import (DataValidationError, add_derived_features, detect_id_column,
                                      generate_synthetic_customers, load_csv, numeric_candidates)
from segmentation.preprocessing import PreprocessConfig, preprocess
from segmentation.schema import detect_roles, normalize_column_name

FEATURES = ["age", "annual_income_lakh", "spending_score", "orders_per_year", "avg_order_value_inr", "days_since_last_order"]


class DataLoaderTests(unittest.TestCase):
    def test_generator_is_reproducible_and_has_defects(self):
        a = generate_synthetic_customers(300, seed=1)
        b = generate_synthetic_customers(300, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertGreater(a.isna().sum().sum(), 0)
        self.assertEqual(a["customer_id"].is_unique, True)

    def test_column_names_normalised(self):
        self.assertEqual(normalize_column_name("Annual Income (k$)"), "annual_income_k")
        self.assertEqual(normalize_column_name("Spending Score (1-100)"), "spending_score_1_100")

    def test_mall_customers_layout_is_supported(self):
        rng = np.random.default_rng(0)
        mall = pd.DataFrame({
            "CustomerID": range(1, 201), "Gender": rng.choice(["Male", "Female"], 200),
            "Age": rng.integers(18, 70, 200), "Annual Income (k$)": rng.integers(15, 137, 200),
            "Spending Score (1-100)": rng.integers(1, 100, 200)})
        buf = io.StringIO(); mall.to_csv(buf, index=False); buf.seek(0)
        df = load_csv(buf)
        self.assertEqual(detect_id_column(df), "customerid")
        self.assertEqual(numeric_candidates(df, "customerid"), ["age", "annual_income_k", "spending_score_1_100"])
        roles = detect_roles(df.columns)
        self.assertEqual(roles["annual_income_k"], "income")
        self.assertEqual(roles["spending_score_1_100"], "spending")

    def test_invalid_files_are_rejected(self):
        with self.assertRaises(DataValidationError):
            load_csv(io.StringIO("a,b\n1,2\n"))            # too few rows
        with self.assertRaises(DataValidationError):
            load_csv(io.StringIO("name,city\n" + "x,y\n" * 30))  # no numeric columns

    def test_age_role_does_not_match_average(self):
        self.assertEqual(detect_roles(["average_order_value", "age"]), {"average_order_value": "basket", "age": "age"})

    def test_derived_spend(self):
        df = add_derived_features(pd.DataFrame({"orders_per_year": [2], "avg_order_value_inr": [1000.0]}))
        self.assertEqual(df["annual_spend_inr"].iloc[0], 2000.0)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic_customers(500, seed=3)

    def cfg(self, **kw):
        return PreprocessConfig(numeric_features=FEATURES, id_column="customer_id", **kw)

    def test_no_missing_values_after_imputation(self):
        res = preprocess(self.df, self.cfg())
        self.assertFalse(np.isnan(res.X).any())
        self.assertEqual(res.X.shape, (500, len(FEATURES)))
        self.assertGreater(sum(res.report["missing_before"].values()), 0)

    def test_standard_scaling_has_zero_mean_unit_variance(self):
        res = preprocess(self.df, self.cfg(scaler="standard", outlier_method="none"))
        np.testing.assert_allclose(res.X.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(res.X.std(axis=0), 1, atol=1e-8)

    def test_minmax_normalisation_range(self):
        res = preprocess(self.df, self.cfg(scaler="minmax"))
        self.assertAlmostEqual(res.X.min(), 0.0)
        self.assertAlmostEqual(res.X.max(), 1.0)

    def test_l2_normalisation_gives_unit_rows(self):
        res = preprocess(self.df, self.cfg(l2_normalize=True))
        np.testing.assert_allclose(np.linalg.norm(res.X, axis=1), 1.0, atol=1e-8)

    def test_outlier_clipping_limits_extremes_but_keeps_original_data(self):
        clipped = preprocess(self.df, self.cfg(outlier_method="iqr_clip"))
        raw = preprocess(self.df, self.cfg(outlier_method="none"))
        self.assertGreater(clipped.report["clipped_values"]["avg_order_value_inr"], 0)
        self.assertLess(clipped.features_raw["avg_order_value_inr"].max(), raw.features_raw["avg_order_value_inr"].max())
        self.assertEqual(clipped.data["avg_order_value_inr"].max(), raw.data["avg_order_value_inr"].max())

    def test_drop_strategy_removes_incomplete_rows(self):
        res = preprocess(self.df, self.cfg(missing_strategy="drop"))
        self.assertLess(len(res.data), len(self.df))
        self.assertEqual(len(res.data), len(res.X))

    def test_duplicates_removed(self):
        doubled = pd.concat([self.df, self.df.head(20)], ignore_index=True)
        res = preprocess(doubled, self.cfg())
        self.assertEqual(res.report["duplicates_removed"], 20)

    def test_categorical_encoding_and_log_transform(self):
        cfg = PreprocessConfig(numeric_features=FEATURES, categorical_features=["gender"], id_column="customer_id",
                               log_transform_skewed=True, skew_threshold=0.5)
        res = preprocess(self.df, cfg)
        self.assertIn("gender_Male", res.feature_names)
        self.assertGreaterEqual(len(res.report["log_transformed"]), 1)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            preprocess(self.df, PreprocessConfig(numeric_features=["age"]))
        with self.assertRaises(ValueError):
            preprocess(self.df, PreprocessConfig(numeric_features=["age", "nope"]))
        with self.assertRaises(ValueError):
            preprocess(self.df, self.cfg(scaler="bogus"))


if __name__ == "__main__":
    unittest.main()
