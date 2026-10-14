"""Data preprocessing: missing values, outliers, skew, scaling and normalisation.

``preprocess`` returns both the model-ready matrix and the cleaned table in original
units (rows aligned), so segments can later be explained in business terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler

from .data_loader import add_derived_features

SCALERS = {
    "standard": StandardScaler,   # z-score standardisation
    "minmax": MinMaxScaler,       # min-max normalisation to [0, 1]
    "robust": RobustScaler,       # median / IQR, resistant to outliers
}
MISSING_STRATEGIES = ("median", "mean", "drop")
OUTLIER_METHODS = ("none", "iqr_clip")


@dataclass
class PreprocessConfig:
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    missing_strategy: str = "median"
    drop_duplicates: bool = True
    outlier_method: str = "iqr_clip"
    iqr_multiplier: float = 1.5
    log_transform_skewed: bool = False
    skew_threshold: float = 1.0
    scaler: str = "standard"
    l2_normalize: bool = False
    id_column: str | None = None

    def validate(self) -> None:
        if len(self.numeric_features) + len(self.categorical_features) < 2:
            raise ValueError("Select at least two features.")
        if self.missing_strategy not in MISSING_STRATEGIES:
            raise ValueError(f"missing_strategy must be one of {MISSING_STRATEGIES}")
        if self.outlier_method not in OUTLIER_METHODS:
            raise ValueError(f"outlier_method must be one of {OUTLIER_METHODS}")
        if self.scaler not in SCALERS:
            raise ValueError(f"scaler must be one of {tuple(SCALERS)}")
        if self.iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive")


@dataclass
class PreprocessResult:
    X: np.ndarray                 # scaled model matrix
    X_df: pd.DataFrame            # same, with column names
    features_raw: pd.DataFrame    # model matrix before scaling (after clip / log)
    data: pd.DataFrame            # cleaned table in original units, aligned with X
    feature_names: list[str]
    report: dict


def preprocess(df: pd.DataFrame, cfg: PreprocessConfig) -> PreprocessResult:
    cfg.validate()
    features = list(cfg.numeric_features) + list(cfg.categorical_features)
    absent = [c for c in features if c not in df.columns]
    if absent:
        raise ValueError(f"Columns not found in data: {absent}")

    data = df.copy()
    report: dict = {"rows_in": int(len(df))}

    # 1. Duplicates (by id when available, otherwise whole-row).
    if cfg.drop_duplicates:
        before = len(data)
        subset = [cfg.id_column] if cfg.id_column and cfg.id_column in data.columns else None
        data = data.drop_duplicates(subset=subset)
        report["duplicates_removed"] = int(before - len(data))
    else:
        report["duplicates_removed"] = 0

    # 2. Numeric coercion; +/-inf are treated as missing.
    for col in cfg.numeric_features:
        data[col] = pd.to_numeric(data[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    report["missing_before"] = {c: int(data[c].isna().sum()) for c in features}

    # 3. Missing values.
    if cfg.missing_strategy == "drop":
        data = data.dropna(subset=features)
    else:
        for col in cfg.numeric_features:
            fill = data[col].median() if cfg.missing_strategy == "median" else data[col].mean()
            data[col] = data[col].fillna(fill)
        for col in cfg.categorical_features:
            mode = data[col].mode(dropna=True)
            data[col] = data[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    data = data.reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f"Only {len(data)} rows left after cleaning; try median imputation.")
    if data[cfg.numeric_features].isna().any().any():
        raise ValueError("A selected numeric column has no valid values.")
    report["rows_out"] = int(len(data))
    report["rows_dropped_for_missing"] = int(report["rows_in"] - report["duplicates_removed"] - len(data))

    data = add_derived_features(data)

    # 4. Outliers: winsorise to the IQR fences (model input only; `data` stays unclipped).
    num = data[cfg.numeric_features].astype(float).copy()
    clipped = {}
    if cfg.outlier_method == "iqr_clip":
        for col in num.columns:
            q1, q3 = num[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                clipped[col] = 0
                continue
            lo, hi = q1 - cfg.iqr_multiplier * iqr, q3 + cfg.iqr_multiplier * iqr
            clipped[col] = int(((num[col] < lo) | (num[col] > hi)).sum())
            num[col] = num[col].clip(lo, hi)
    report["clipped_values"] = clipped

    # 5. Optional log1p for right-skewed, non-negative columns.
    logged = []
    if cfg.log_transform_skewed:
        for col in num.columns:
            if num[col].min() >= 0 and num[col].skew() > cfg.skew_threshold:
                num[col] = np.log1p(num[col])
                logged.append(col)
    report["log_transformed"] = logged

    # 6. Categorical encoding (one-hot; binary columns keep a single indicator).
    parts = [num]
    for col in cfg.categorical_features:
        dummies = pd.get_dummies(data[col].astype(str), prefix=col,
                                 drop_first=data[col].nunique() == 2, dtype=float)
        parts.append(dummies)
    raw = pd.concat(parts, axis=1)

    # 7. Scaling, then optional row-wise L2 normalisation.
    X = SCALERS[cfg.scaler]().fit_transform(raw.values)
    if cfg.l2_normalize:
        X = Normalizer(norm="l2").fit_transform(X)

    report.update(scaler=cfg.scaler, l2_normalize=cfg.l2_normalize,
                  missing_strategy=cfg.missing_strategy, outlier_method=cfg.outlier_method,
                  features_used=list(raw.columns))
    return PreprocessResult(
        X=X,
        X_df=pd.DataFrame(X, columns=raw.columns),
        features_raw=raw,
        data=data,
        feature_names=list(raw.columns),
        report=report,
    )
