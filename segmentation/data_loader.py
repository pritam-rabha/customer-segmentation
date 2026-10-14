"""Loading, validating and generating customer data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, SAMPLE_DATA_PATH
from .schema import normalize_column_name


class DataValidationError(ValueError):
    """Raised when an input file cannot be used for segmentation."""


# --------------------------------------------------------------------------- loading
def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that are >=90% numeric into numbers (stray text -> NaN)."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].astype("string").str.strip().replace("", pd.NA)
        non_null = series.dropna()
        if non_null.empty:
            df[col] = series
            continue
        converted = pd.to_numeric(non_null.str.replace(",", "", regex=False), errors="coerce")
        if converted.notna().mean() >= 0.9:
            df[col] = pd.to_numeric(series.str.replace(",", "", regex=False), errors="coerce")
        else:
            df[col] = series
    return df


def load_csv(source, min_rows: int = 20) -> pd.DataFrame:
    """Read a CSV (path or file-like), normalise headers and validate it."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pandas raises many different error types
        raise DataValidationError(f"Could not read the CSV file: {exc}") from exc

    df.columns = [normalize_column_name(c) for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise DataValidationError("Column names are not unique after normalisation.")
    df = _coerce_numeric_like(df)

    if len(df) < min_rows:
        raise DataValidationError(f"Need at least {min_rows} rows; the file has {len(df)}.")
    if len(numeric_candidates(df, detect_id_column(df))) < 2:
        raise DataValidationError("Need at least two numeric columns to segment customers.")
    return df


def load_sample(path: Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Load the bundled sample, regenerating it if the file is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_customers().to_csv(path, index=False)
    return load_csv(path)


def detect_id_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col == "id" or col.endswith("_id") or (col.endswith("id") and "customer" in col):
            return col
    return None


def numeric_candidates(df: pd.DataFrame, id_col: str | None = None) -> list[str]:
    return [
        c for c in df.columns
        if c != id_col and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]


def categorical_candidates(df: pd.DataFrame, id_col: str | None = None, max_levels: int = 10) -> list[str]:
    return [
        c for c in df.columns
        if c != id_col and not pd.api.types.is_numeric_dtype(df[c])
        and 1 < df[c].nunique(dropna=True) <= max_levels
    ]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimated annual spend (orders x basket) when both columns exist."""
    df = df.copy()
    if "orders_per_year" not in df.columns:
        return df
    for basket, out in (("avg_order_value_inr", "annual_spend_inr"), ("avg_order_value", "annual_spend")):
        if basket in df.columns and out not in df.columns:
            df[out] = (df["orders_per_year"] * df[basket]).round(0)
            break
    return df


# ----------------------------------------------------------------------- generation
_TIER_CITIES = {
    "Tier 1": ([("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bengaluru", "Karnataka"), ("Hyderabad", "Telangana"),
                ("Chennai", "Tamil Nadu"), ("Kolkata", "West Bengal"), ("Pune", "Maharashtra"), ("Ahmedabad", "Gujarat")],
               [5, 5, 5, 4, 3, 3, 3, 2]),
    "Tier 2": ([("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh"), ("Indore", "Madhya Pradesh"), ("Bhopal", "Madhya Pradesh"),
                ("Nagpur", "Maharashtra"), ("Chandigarh", "Chandigarh"), ("Kochi", "Kerala"), ("Guwahati", "Assam"),
                ("Surat", "Gujarat"), ("Patna", "Bihar")], [1] * 10),
    "Tier 3": ([("Gwalior", "Madhya Pradesh"), ("Sagar", "Madhya Pradesh"), ("Jabalpur", "Madhya Pradesh"),
                ("Ujjain", "Madhya Pradesh"), ("Udaipur", "Rajasthan"), ("Mysuru", "Karnataka"),
                ("Dehradun", "Uttarakhand"), ("Siliguri", "West Bengal")], [1] * 8),
}
_PAYMENTS = ["UPI", "Credit/Debit card", "Cash on delivery", "Wallet"]
_CATEGORIES = ["Fashion", "Electronics", "Grocery", "Home & Kitchen", "Beauty & Personal Care"]

_MALE = ["Aarav", "Vivaan", "Arjun", "Rohan", "Amit", "Rahul", "Sandeep", "Vikram", "Suresh", "Manoj", "Karan", "Ankit",
         "Pranav", "Deepak", "Nikhil", "Rajesh", "Sanjay", "Harsh", "Mohit", "Gaurav", "Siddharth", "Abhishek"]
_FEMALE = ["Ananya", "Priya", "Neha", "Pooja", "Sneha", "Kavya", "Isha", "Riya", "Anjali", "Meera", "Divya", "Shreya",
           "Aditi", "Swati", "Nisha", "Komal", "Ritu", "Sunita", "Lakshmi", "Rekha", "Tanvi", "Pallavi"]
_SURNAMES = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Reddy", "Iyer", "Nair", "Das", "Banerjee", "Mehta", "Joshi",
             "Kulkarni", "Rao", "Chauhan", "Yadav", "Mishra", "Bora", "Kalita", "Thakur", "Pandey", "Agarwal", "Jain",
             "Kapoor", "Malhotra", "Shah", "Desai", "Menon", "Pillai", "Chatterjee", "Ghosh", "Tiwari", "Saxena"]

# Latent behavioural groups: (mean, std) per attribute, plus categorical mixes. The groups
# overlap on purpose so clustering is realistic rather than trivially perfect.
_GROUPS = {
    "champions": dict(w=.18, age=(38, 8), income=(18, 4.5), spend=(80, 9), tenure=(48, 14), orders=(36, 7), aov=(2400, 500),
                      recency=(12, 8), online=(62, 15), tier=(.55, .33, .12), pay=(.45, .45, .05, .05),
                      cat=(.25, .30, .15, .15, .15)),
    "affluent_quiet": dict(w=.14, age=(47, 9), income=(24, 5), spend=(26, 9), tenure=(30, 12), orders=(8, 3), aov=(3600, 800),
                           recency=(70, 25), online=(40, 15), tier=(.50, .35, .15), pay=(.35, .55, .03, .07),
                           cat=(.20, .35, .10, .25, .10)),
    "budget_fans": dict(w=.22, age=(26, 6), income=(5, 1.8), spend=(78, 10), tenure=(20, 9), orders=(30, 6), aov=(650, 200),
                        recency=(18, 10), online=(78, 12), tier=(.20, .40, .40), pay=(.60, .05, .20, .15),
                        cat=(.35, .10, .25, .10, .20)),
    "mainstream": dict(w=.30, age=(41, 10), income=(9, 2.5), spend=(50, 11), tenure=(28, 12), orders=(15, 5), aov=(1300, 350),
                       recency=(35, 15), online=(50, 18), tier=(.30, .40, .30), pay=(.50, .25, .15, .10),
                       cat=(.20, .15, .30, .20, .15)),
    "dormant": dict(w=.16, age=(53, 10), income=(5.5, 2), spend=(21, 8), tenure=(40, 15), orders=(4, 2), aov=(750, 250),
                    recency=(165, 45), online=(30, 15), tier=(.25, .35, .40), pay=(.35, .15, .40, .10),
                    cat=(.25, .10, .30, .20, .15)),
}

_BOUNDS = dict(age=(18, 75), income=(1.5, 60), spend=(1, 100), tenure=(1, 120),
               orders=(1, 80), aov=(150, 15000), recency=(1, 365), online=(0, 100))


def generate_synthetic_customers(n: int = 1000, seed: int = RANDOM_STATE,
                                 missing_rate: float = 0.04, outlier_rate: float = 0.015,
                                 return_truth: bool = False):
    """Create a reproducible Indian retail / e-commerce customer table.

    Amounts are in rupees (income in lakh per year, basket in rupees). The table has
    realistic defects (missing values, a few extreme baskets) so preprocessing has real
    work to do. With ``return_truth`` the latent group of every row is also returned
    (used by the tests, never written to the CSV).
    """
    rng = np.random.default_rng(seed)
    names = list(_GROUPS)
    weights = np.array([_GROUPS[g]["w"] for g in names])
    group_idx = rng.choice(len(names), size=n, p=weights / weights.sum())

    cols = {k: np.zeros(n) for k in _BOUNDS}
    tier = np.empty(n, dtype=object)
    pay = np.empty(n, dtype=object)
    cat = np.empty(n, dtype=object)
    for gi, g in enumerate(names):
        mask = group_idx == gi
        m = int(mask.sum())
        spec = _GROUPS[g]
        for attr in cols:
            mean, std = spec[attr]
            cols[attr][mask] = rng.normal(mean, std, m)
        tier[mask] = rng.choice(list(_TIER_CITIES), size=m, p=spec["tier"])
        pay[mask] = rng.choice(_PAYMENTS, size=m, p=spec["pay"])
        cat[mask] = rng.choice(_CATEGORIES, size=m, p=spec["cat"])
    for attr, (lo, hi) in _BOUNDS.items():
        cols[attr] = np.clip(cols[attr], lo, hi)

    city, state = np.empty(n, dtype=object), np.empty(n, dtype=object)
    for t, (places, w) in _TIER_CITIES.items():
        idx = np.where(tier == t)[0]
        pick = rng.choice(len(places), size=len(idx), p=np.array(w) / sum(w))
        city[idx] = [places[i][0] for i in pick]
        state[idx] = [places[i][1] for i in pick]

    gender = rng.choice(["Female", "Male"], size=n, p=[.48, .52])
    person = [f"{rng.choice(_FEMALE if g == 'Female' else _MALE)} {rng.choice(_SURNAMES)}" for g in gender]

    df = pd.DataFrame({
        "customer_id": [f"IN{i + 1:05d}" for i in range(n)],
        "customer_name": person,
        "gender": gender,
        "age": cols["age"].round().astype(int),
        "city": city,
        "state": state,
        "city_tier": tier,
        "annual_income_lakh": cols["income"].round(1),
        "spending_score": cols["spend"].round().astype(int),
        "tenure_months": cols["tenure"].round().astype(int),
        "orders_per_year": cols["orders"].round().astype(int),
        "avg_order_value_inr": (np.round(cols["aov"] / 10) * 10),
        "days_since_last_order": cols["recency"].round().astype(int),
        "online_share_pct": cols["online"].round(1),
        "preferred_payment": pay,
        "top_category": cat,
    })

    # A handful of extreme customers (bulk / corporate buyers or entry slips).
    n_out = int(round(outlier_rate * n))
    out_rows = rng.choice(n, size=n_out, replace=False)
    half = n_out // 2
    df.loc[out_rows[:half], "avg_order_value_inr"] = df.loc[out_rows[:half], "avg_order_value_inr"] * 3.5
    df.loc[out_rows[half:], "orders_per_year"] = (df.loc[out_rows[half:], "orders_per_year"] * 2.5).round().astype(int)

    # Missing values in a few columns.
    df = df.astype({"age": float, "tenure_months": float})
    for col in ["age", "annual_income_lakh", "tenure_months", "avg_order_value_inr", "online_share_pct", "gender"]:
        df.loc[rng.random(n) < missing_rate, col] = np.nan

    return (df, np.array(names)[group_idx]) if return_truth else df
