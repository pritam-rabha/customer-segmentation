"""Turn cluster labels into business-ready segment profiles and recommendations.

Segments are named from two data-driven axes computed per cluster as z-scores against
the whole customer base:

    value       - income, basket size, estimated annual spend (higher = more valuable)
    engagement  - spending score, order frequency, recency (fewer days since the last
                  order counts as *more* engaged)

Datasets without recognisable columns fall back to neutral "Segment n" names plus the
traits that distinguish each cluster most.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clustering import NOISE
from .formatting import format_value
from .schema import detect_roles, display_name, short_name

NOISE_NAME = "Unassigned (noise)"
HIGH, LOW = 0.4, -0.4
VALUE_ROLES = ("income", "basket", "monetary")
ENGAGEMENT_ROLES = ("spending", "frequency", "recency")

PLAYBOOK = {
    "Champions": (
        "High-value customers who buy often and recently.",
        ["Offer a VIP or early-access tier so loyalty is recognised before it is requested.",
         "Ask for reviews and referrals; they are the most credible advocates.",
         "Protect margin: use exclusives and service perks instead of blanket discounts."],
        "Retention rate and share of revenue from this segment",
    ),
    "Affluent, low engagement": (
        "Strong spending capacity but little activity; the largest untapped upside.",
        ["Run a personalised re-engagement campaign with premium product recommendations.",
         "Test concierge service or curated bundles that suit large baskets.",
         "Find the preferred channel and cadence before increasing contact frequency."],
        "Orders per year and reactivation rate",
    ),
    "Premium regulars": (
        "Above-average value with steady engagement.",
        ["Create a clear upgrade path in the loyalty programme.",
         "Cross-sell premium categories and offer subscription or replenishment options.",
         "Track basket growth after each upgrade offer."],
        "Average order value and tier-upgrade rate",
    ),
    "Enthusiastic budget shoppers": (
        "Frequent, engaged buyers with small baskets.",
        ["Raise basket size with bundles and multi-buy offers rather than deeper discounts.",
         "Reward frequency with points that unlock higher-value products.",
         "Recommend trade-up items next to their usual purchases."],
        "Average order value",
    ),
    "Dormant / at risk": (
        "Low value and low engagement; most have not ordered for a long time.",
        ["Send a time-limited win-back offer, then a short survey asking why they stopped.",
         "Cap spend on this group: stop paid outreach after the final attempt.",
         "Review whether a product or service issue is driving lapse."],
        "Win-back rate and cost per reactivated customer",
    ),
    "Price-sensitive occasionals": (
        "Low value, moderate engagement; buy when prices or events trigger them.",
        ["Tie promotions to seasonal moments and use low-cost channels such as email.",
         "Nudge second and third purchases with small, time-boxed incentives.",
         "Measure promotion uplift against a holdout group to avoid paying for sales that would happen anyway."],
        "Repeat-purchase rate",
    ),
    "Engaged mainstream": (
        "Typical value with above-average activity.",
        ["Cross-sell adjacent categories based on purchase history.",
         "Add progress-based loyalty rewards and referral incentives.",
         "Watch for movement into the Champions group."],
        "Orders per year",
    ),
    "Drifting mainstream": (
        "Typical value but activity is slipping.",
        ["Trigger reminders around each customer's usual reorder cycle.",
         "Reach out before the lapse becomes permanent; early nudges are cheaper than win-back.",
         "Collect quick feedback on service and range."],
        "Days since last order",
    ),
    "Core mainstream": (
        "Average value and average engagement; the stable base.",
        ["Maintain with lifecycle emails and relevant recommendations.",
         "Test upsell offers on small samples first.",
         "Monitor movement between segments each month."],
        "Segment migration rate",
    ),
    NOISE_NAME: (
        "Customers whose combination of traits does not fit any dense group.",
        ["Review individually: some may be high-potential accounts, others data errors.",
         "Exclude them from average-based targeting rules.",
         "Re-check after tuning eps and min_samples."],
        "Share of customers left unassigned",
    ),
    "Segment": (
        "Group defined by the distinguishing traits listed below.",
        ["Test one targeted offer aligned to the strongest trait.",
         "Keep a holdout group so uplift can be measured.",
         "Re-evaluate after the next data refresh."],
        "Campaign response rate",
    ),
}


@dataclass
class SegmentProfile:
    means: pd.DataFrame          # cluster id x numeric columns (original units)
    z: pd.DataFrame              # means expressed in overall standard deviations
    sizes: pd.Series
    share_pct: pd.Series
    spend_share_pct: pd.Series | None
    overall_mean: pd.Series
    roles: dict = field(default_factory=dict)


@dataclass
class SegmentInsight:
    cluster_id: int
    name: str
    archetype: str
    customers: int
    share_pct: float
    spend_share_pct: float | None
    summary: str
    traits: list[str]
    actions: list[str]
    kpi: str


def _ordered_ids(ids) -> list[int]:
    return sorted((i for i in ids if i != NOISE)) + ([NOISE] if NOISE in set(ids) else [])


def profile_segments(data: pd.DataFrame, labels, id_column: str | None = None) -> SegmentProfile:
    labels = np.asarray(labels)
    if len(labels) != len(data):
        raise ValueError("labels and data must have the same length")
    df = data.copy()
    df["__cluster__"] = labels
    cols = [c for c in df.select_dtypes(include="number").columns if c not in (id_column, "__cluster__")]
    ids = _ordered_ids(df["__cluster__"].unique())

    means = df.groupby("__cluster__")[cols].mean().reindex(ids)
    overall_mean = df[cols].mean()
    overall_std = df[cols].std(ddof=0).replace(0, np.nan)
    z = ((means - overall_mean) / overall_std).fillna(0.0)

    sizes = df["__cluster__"].value_counts().reindex(ids)
    share = sizes / len(df) * 100
    spend_share = None
    roles = detect_roles(cols)
    spend_col = next((c for c, r in roles.items() if r == "monetary" and c.startswith("annual_spend")), None)
    if spend_col and df[spend_col].sum() > 0:
        spend_share = df.groupby("__cluster__")[spend_col].sum().reindex(ids) / df[spend_col].sum() * 100

    return SegmentProfile(means, z, sizes, share, spend_share, overall_mean, roles)


def _axis_scores(z_row: pd.Series, roles: dict) -> tuple[float | None, float | None]:
    value, engagement = [], []
    for col, role in roles.items():
        if col not in z_row.index:
            continue
        if role in VALUE_ROLES:
            value.append(z_row[col])
        elif role in ENGAGEMENT_ROLES:
            engagement.append(-z_row[col] if role == "recency" else z_row[col])
    return (float(np.mean(value)) if value else None,
            float(np.mean(engagement)) if engagement else None)


def classify_segment(value: float | None, engagement: float | None) -> str:
    """Map the two axis scores onto a named archetype."""
    if value is None and engagement is None:
        return "Segment"
    v = 0.0 if value is None else value
    e = 0.0 if engagement is None else engagement
    if v >= HIGH:
        return "Champions" if e >= HIGH else "Affluent, low engagement" if e <= LOW else "Premium regulars"
    if v <= LOW:
        return "Enthusiastic budget shoppers" if e >= HIGH else "Dormant / at risk" if e <= LOW else "Price-sensitive occasionals"
    return "Engaged mainstream" if e >= HIGH else "Drifting mainstream" if e <= LOW else "Core mainstream"


def _traits(profile: SegmentProfile, cid: int, feature_cols: list[str], top: int = 3) -> list[str]:
    cols = [c for c in feature_cols if c in profile.z.columns]
    ranked = profile.z.loc[cid, cols].abs().sort_values(ascending=False).index[:top]
    lines = []
    for col in ranked:
        mean, overall = profile.means.loc[cid, col], profile.overall_mean[col]
        pct = f" ({(mean / overall - 1) * 100:+.0f}%)" if overall else ""
        lines.append(f"{short_name(col)}: {format_value(col, mean)} vs {format_value(col, overall)} overall{pct}")
    return lines


def build_insights(profile: SegmentProfile, feature_cols: list[str]) -> list[SegmentInsight]:
    """One insight per cluster, ordered by cluster id (noise last)."""
    archetypes = {}
    for cid in profile.means.index:
        if cid == NOISE:
            archetypes[cid] = NOISE_NAME
        else:
            archetypes[cid] = classify_segment(*_axis_scores(profile.z.loc[cid], profile.roles))

    counts = pd.Series(archetypes).value_counts()
    insights = []
    for n, cid in enumerate(profile.means.index, start=1):
        arch = archetypes[cid]
        if cid == NOISE:
            name = NOISE_NAME
        elif arch == "Segment":
            name = f"Segment {int(cid) + 1}"
        elif counts[arch] > 1:
            name = f"{arch} (cluster {int(cid)})"
        else:
            name = arch
        summary, actions, kpi = PLAYBOOK[arch]
        insights.append(SegmentInsight(
            cluster_id=int(cid), name=name, archetype=arch,
            customers=int(profile.sizes[cid]), share_pct=float(profile.share_pct[cid]),
            spend_share_pct=None if profile.spend_share_pct is None else float(profile.spend_share_pct[cid]),
            summary=summary,
            traits=_traits(profile, cid, feature_cols),
            actions=list(actions), kpi=kpi,
        ))
    return insights


def segment_name_map(insights: list[SegmentInsight]) -> dict[int, str]:
    return {i.cluster_id: i.name for i in insights}


def profile_table(profile: SegmentProfile, insights: list[SegmentInsight]) -> pd.DataFrame:
    """Display-ready table: size, share, spend share and mean of every metric."""
    names = segment_name_map(insights)
    table = pd.DataFrame({
        "Segment": [names[i] for i in profile.means.index],
        "Customers": profile.sizes.values.astype(int),
        "Share of customers (%)": profile.share_pct.round(1).values,
    })
    if profile.spend_share_pct is not None:
        table["Share of est. spend (%)"] = profile.spend_share_pct.round(1).values
    means = profile.means.round(1).copy()
    means.columns = [display_name(c) for c in means.columns]
    return pd.concat([table, means.reset_index(drop=True)], axis=1)


def insights_to_markdown(insights: list[SegmentInsight], title: str = "Customer segmentation report",
                         context: dict | None = None) -> str:
    lines = [f"# {title}", ""]
    for key, value in (context or {}).items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")
    for ins in insights:
        lines += [f"## {ins.name}", "", ins.summary, "",
                  f"- Customers: {ins.customers:,} ({ins.share_pct:.1f}% of base)"]
        if ins.spend_share_pct is not None:
            lines.append(f"- Share of estimated annual spend: {ins.spend_share_pct:.1f}%")
        lines += ["", "**What sets them apart**", ""] + [f"- {t}" for t in ins.traits]
        lines += ["", "**Recommended actions**", ""] + [f"- {a}" for a in ins.actions]
        lines += ["", f"**Metric to watch:** {ins.kpi}", ""]
    return "\n".join(lines)
