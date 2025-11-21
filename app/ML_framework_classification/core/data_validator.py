# app/ML_framework_classification/core/data_validator.py
from fastapi import HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
from .session import session
import pandas as pd

class ValidationErrorDetail(BaseModel):
    check: str
    passed: bool
    message: str
    severity: str  # "error" or "warning"

class DataValidationReport(BaseModel):
    overall_passed: bool
    errors: List[ValidationErrorDetail]
    warnings: List[ValidationErrorDetail]
    summary: str

def validate_dataset() -> DataValidationReport:
    if session.df is None or session.X is None or session.y is None:
        raise HTTPException(status_code=400, detail="Run upload + column selection first")

    df = session.df
    X = session.X
    y = session.y
    X_cols = session.X_cols
    y_col = session.y_col

    errors: List[ValidationErrorDetail] = []
    warnings: List[ValidationErrorDetail] = []

    # === 1. Empty dataset ===
    if df.empty:
        errors.append(ValidationErrorDetail(
            check="empty_dataset",
            passed=False,
            message="Dataset is empty",
            severity="error"
        ))

    # === 2. Selected columns exist (already checked, but double-confirm) ===
    missing_cols = [c for c in X_cols + [y_col] if c not in df.columns]
    if missing_cols:
        errors.append(ValidationErrorDetail(
            check="missing_columns",
            passed=False,
            message=f"Columns not found in dataset: {missing_cols}",
            severity="error"
        ))

    # === 3. Duplicate columns in X ===
    if len(X_cols) != len(set(X_cols)):
        dupes = [item for item in set(X_cols) if X_cols.count(item) > 1]
        errors.append(ValidationErrorDetail(
            check="duplicate_features",
            passed=False,
            message=f"Duplicate columns in features: {dupes}",
            severity="error"
        ))

    # === 4. Target has ≥ 2 classes ===
    n_classes = y.nunique()
    if n_classes < 2:
        errors.append(ValidationErrorDetail(
            check="target_classes_count",
            passed=False,
            message=f"Target column has only {n_classes} class. Need ≥2 for classification.",
            severity="error"
        ))

    # === 5. Target is not constant ===
    if y.nunique() == 1:
        errors.append(ValidationErrorDetail(
            check="constant_target",
            passed=False,
            message=f"Target column '{y_col}' is constant (all values same). No learning possible.",
            severity="error"
        ))

    # === 6. Target is not all unique (ID leakage risk) ===
    if y.nunique() == len(y):
        errors.append(ValidationErrorDetail(
            check="target_all_unique",
            passed=False,
            message=f"Target column has all unique values. Likely an ID column, not a label.",
            severity="error"
        ))

    # === 7. Constant features (warning) ===
    constant_features = [col for col in X_cols if X[col].nunique() <= 1]
    if constant_features:
        warnings.append(ValidationErrorDetail(
            check="constant_features",
            passed=True,
            message=f"Constant features detected (will be dropped): {constant_features}",
            severity="warning"
        ))

    # === 8. Near-constant features (99% same value) ===
    quasi_constant = []
    for col in X_cols:
        if X[col].dtype == "object":
            top_freq = X[col].value_counts().iloc[0] / len(X[col])
            if top_freq > 0.99:
                quasi_constant.append(col)
        else:
            if X[col].std() == 0:
                quasi_constant.append(col)
    if quasi_constant:
        warnings.append(ValidationErrorDetail(
            check="quasi_constant_features",
            passed=True,
            message=f"Quasi-constant features (>99% same value): {quasi_constant[:5]}",
            severity="warning"
        ))

    # Final result
    overall_passed = len(errors) == 0

    summary = "All checks passed!" if overall_passed else f"{len(errors)} critical error(s) found."

    return DataValidationReport(
        overall_passed=overall_passed,
        errors=errors,
        warnings=warnings,
        summary=summary
    )