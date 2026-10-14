import unittest

from segmentation.data_loader import generate_synthetic_customers
from segmentation.formatting import format_inr, format_value, indian_grouping


class FormattingTests(unittest.TestCase):
    def test_indian_digit_grouping(self):
        self.assertEqual(indian_grouping(1234567), "12,34,567")
        self.assertEqual(indian_grouping(999), "999")
        self.assertEqual(indian_grouping(100000), "1,00,000")

    def test_rupee_formats(self):
        self.assertEqual(format_inr(3512.4), "₹3,512")
        self.assertEqual(format_inr(1234567), "₹12.35 lakh")
        self.assertEqual(format_inr(32100000), "₹3.21 Cr")
        self.assertEqual(format_inr(1234567, compact=False), "₹12,34,567")
        self.assertEqual(format_inr(float("nan")), "n/a")

    def test_value_formats_follow_column_meaning(self):
        self.assertEqual(format_value("annual_income_lakh", 18.24), "₹18.2 lakh")
        self.assertEqual(format_value("days_since_last_order", 12.4), "12 days")
        self.assertEqual(format_value("online_share_pct", 61.6), "62%")


class IndianSampleTests(unittest.TestCase):
    def test_sample_is_indian(self):
        df = generate_synthetic_customers(600, seed=5)
        self.assertEqual(set(df["city_tier"].dropna()), {"Tier 1", "Tier 2", "Tier 3"})
        self.assertIn("Mumbai", set(df["city"]))
        self.assertIn("UPI", set(df["preferred_payment"]))
        self.assertTrue(df["customer_id"].str.startswith("IN").all())
        self.assertTrue(df["annual_income_lakh"].dropna().between(1.5, 60).all())
        self.assertTrue(df.groupby("city")["state"].nunique().le(1).all())  # each city maps to one state


if __name__ == "__main__":
    unittest.main()
