# app/ML_framework_classification/core/data_analyzer.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from .session import session
from ..schemas import FeatureInfo, TargetAnalysis, DataUnderstandingReport

def infer_feature_type(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col) or col.dtype == bool:
        return "boolean"
    if pd.api.types.is_numeric_dtype(col):
        unique_ratio = col.nunique() / len(col)
        if unique_ratio > 0.95 and col.nunique() > 20:
            return "numerical"  # high cardinality numeric → treat as continuous
        elif col.nunique() <= 20:
            return "categorical"
        else:
            return "numerical"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    if col.dtype == object:
        sample = col.dropna().astype(str).str.strip()
        if col.nunique() < 50 or sample.str.contains(r'^\d{4}-\d{2}-\d{2}', regex=True).any():
            return "categorical"
        if sample.str.len().mean() > 50:
            return "text"
        return "categorical"
    return "categorical"

def analyze_dataset() -> DataUnderstandingReport:
    if session.df is None or session.X is None or session.y is None:
        raise ValueError("Run upload + column selection first")

    df = session.df
    X_cols = session.X_cols
    y_col = session.y_col

    features: List[FeatureInfo] = []
    quality_flags = []
    recommendations = []

    # Analyze each feature
    for col_name in X_cols:
        col = df[col_name]
        missing_count = col.isna().sum()
        missing_pct = missing_count / len(df)
        unique_count = col.nunique()

        info = FeatureInfo(
            name=col_name,
            inferred_type=infer_feature_type(col),
            dtype=str(col.dtype),
            missing_count=int(missing_count),
            missing_pct=round(float(missing_pct * 100), 2),
            unique_count=int(unique_count),
            cardinality_pct=round(unique_count / len(df) * 100, 2),
            sample_values=col.dropna().head(5).tolist(),
            is_target=False
        )
        features.append(info)

        # Flags
        if missing_pct > 0.3:
            quality_flags.append(f"High missing values in '{col_name}' ({info.missing_pct}%)")
        if info.inferred_type == "categorical" and unique_count > 50:
            quality_flags.append(f"High cardinality categorical: '{col_name}' ({unique_count} unique)")

    # Target Analysis
    y = session.y
    class_counts = y.value_counts()
    distribution = {str(k): int(v) for k, v in class_counts.to_dict().items()}
    classes = list(distribution.keys())
    n_classes = len(classes)

    imbalance_ratio = None
    is_imbalanced = False
    if n_classes >= 2:
        counts = list(distribution.values())
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 3.0:
            is_imbalanced = True
            quality_flags.append(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")

    target_analysis = TargetAnalysis(
        classes=classes,
        distribution=distribution,
        n_classes=n_classes,
        imbalance_ratio=round(imbalance_ratio, 2) if imbalance_ratio else None,
        is_imbalanced=is_imbalanced,
        recommendation="Consider SMOTE or class weights" if is_imbalanced else None
    )

    # Global recommendations
    missing_cols = [f.name for f in features if f.missing_pct > 0]
    if missing_cols:
        recommendations.append(f"Impute or drop columns with missing values: {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}")

    high_card_cols = [f.name for f in features if f.inferred_type == "categorical" and f.unique_count > 30]
    if high_card_cols:
        recommendations.append(f"Apply target encoding or frequency encoding to high-cardinality cols: {high_card_cols[:3]}")

    return DataUnderstandingReport(
        total_rows=len(df),
        total_columns=len(df.columns),
        features=features,
        target=target_analysis,
        quality_flags=quality_flags,
        recommendations=recommendations,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    )