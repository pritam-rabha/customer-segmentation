"""Column-name normalisation and business-role detection.

Real customer files name columns differently ("Annual Income (k$)", "AOV", ...).
Roles let the insight engine reason about *what a column means* (value, engagement,
recency ...) without hard-coding a single dataset.
"""
from __future__ import annotations

import re

# Ordered: the first matching role wins. Keywords of <=3 characters must match a whole
# "_"-separated token so that e.g. "age" does not match "average".
ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("monetary", ("annual_spend", "total_spend", "revenue", "monetary", "lifetime_value", "clv")),
    ("income", ("income", "salary")),
    ("spending", ("spending",)),
    ("recency", ("days_since", "recency", "last_order", "last_purchase")),
    ("frequency", ("orders", "frequency", "purchases", "visits", "transactions")),
    ("basket", ("order_value", "aov", "basket", "avg_order", "average_order", "amount")),
    ("tenure", ("tenure", "membership", "customer_since")),
    ("age", ("age",)),
    ("channel", ("online", "digital")),
]

DISPLAY_NAMES = {
    "age": "Age",
    "annual_income_lakh": "Annual income (₹ lakh)",
    "annual_income_k": "Annual income (k$)",
    "spending_score": "Spending score (1-100)",
    "tenure_months": "Tenure (months)",
    "orders_per_year": "Orders per year",
    "avg_order_value_inr": "Avg order value (₹)",
    "avg_order_value": "Avg order value",
    "days_since_last_order": "Days since last order",
    "online_share_pct": "Online share (%)",
    "annual_spend_inr": "Est. annual spend (₹)",
    "annual_spend": "Est. annual spend",
    "city_tier": "City tier",
    "preferred_payment": "Preferred payment",
    "top_category": "Top category",
    "customer_name": "Customer",
    "state": "State",
    "city": "City",
    "gender": "Gender",
}


def normalize_column_name(name) -> str:
    """'Annual Income (k$)' -> 'annual_income_k'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def display_name(column: str) -> str:
    """Human-friendly label for a normalised column name."""
    if column in DISPLAY_NAMES:
        return DISPLAY_NAMES[column]
    text = column.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def _matches(keyword: str, column: str) -> bool:
    if len(keyword) <= 3:
        return keyword in column.split("_")
    return keyword in column


def detect_role(column: str) -> str | None:
    for role, keywords in ROLE_KEYWORDS:
        if any(_matches(k, column) for k in keywords):
            return role
    return None


def detect_roles(columns) -> dict[str, str]:
    """Map column name -> role for every column whose role can be recognised."""
    roles = {}
    for col in columns:
        role = detect_role(col)
        if role:
            roles[col] = role
    return roles


def short_name(column: str) -> str:
    """Display name without the unit suffix: 'Avg order value (₹)' -> 'Avg order value'."""
    return display_name(column).split(" (")[0]
