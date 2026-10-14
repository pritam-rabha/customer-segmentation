"""Indian-locale number formatting (rupees, lakh / crore, 12,34,567 digit grouping)."""
from __future__ import annotations

import math


def indian_grouping(n: int) -> str:
    """1234567 -> '12,34,567'."""
    s = str(abs(int(n)))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


def format_inr(value, compact: bool = True) -> str:
    """Rupee amount. Compact form switches to lakh (1e5) and crore (1e7)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    if compact and v >= 1e7:
        return f"{sign}₹{v / 1e7:.2f} Cr"
    if compact and v >= 1e5:
        return f"{sign}₹{v / 1e5:.2f} lakh"
    return f"{sign}₹{indian_grouping(round(v))}"


def format_value(column: str, value) -> str:
    """Format a metric according to what the column measures."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if column.endswith("_inr") or column.startswith("annual_spend"):
        return format_inr(value)
    if "lakh" in column:
        return f"₹{value:.1f} lakh"
    if column.endswith("_pct"):
        return f"{value:.0f}%"
    if "days" in column:
        return f"{value:.0f} days"
    if "months" in column:
        return f"{value:.0f} months"
    if column == "age":
        return f"{value:.0f} yrs"
    return f"{value:,.1f}"
