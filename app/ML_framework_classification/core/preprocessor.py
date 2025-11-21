# app/ML_framework_classification/core/preprocessor.py
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .session import session
from .data_analyzer import infer_feature_type
from sklearn.preprocessing import FunctionTransformer

class PreprocessingResult(BaseModel):
    pipeline_built: bool
    n_features_in: int
    n_features_out: int
    transformed_preview: List[Dict[str, Any]]
    feature_names_out: List[str]
    pipeline_summary: Dict[str, List[str]]
    message: str

def build_preprocessing_pipeline() -> PreprocessingResult:
    if session.X is None or session.y is None:
        raise ValueError("Run column selection first")

    X = session.X.copy()
    feature_types = {}

    # Re-infer types (using same logic as Part 2)
    for col in X.columns:
        feature_types[col] = infer_feature_type(X[col])

    # === Define pipelines per type ===
    numerical_features = [
        col for col in X.columns
        if feature_types[col] == "numerical"
    ]
    categorical_features = [
        col for col in X.columns
        if feature_types[col] in ("categorical", "boolean")
    ]
    datetime_features = [
        col for col in X.columns
        if feature_types[col] == "datetime"
    ]
    text_features = [
        col for col in X.columns
        if feature_types[col] == "text" and X[col].notna().sum() > 10
    ]

    transformers = []

    # Numerical pipeline
    if numerical_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_features))

    # Categorical pipeline
    if categorical_features:
        # Auto choose: OneHot if low cardinality, Ordinal if high
        low_card_cols = [c for c in categorical_features if X[c].nunique() <= 10]
        high_card_cols = [c for c in categorical_features if X[c].nunique() > 10]

        if low_card_cols:
            cat_low = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat_low", cat_low, low_card_cols))

        if high_card_cols:
            cat_high = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            transformers.append(("cat_high", cat_high, high_card_cols))

    # Datetime features
    if datetime_features:
        def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
            for col in datetime_features:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = dt.dt.year.fillna(0)
                df[f"{col}_month"] = dt.dt.month.fillna(0)
                df[f"{col}_day"] = dt.dt.day.fillna(0)
                df[f"{col}_weekday"] = dt.dt.weekday.fillna(0)
                df[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
            return df.drop(columns=datetime_features, errors="ignore")

        datetime_transformer = Pipeline([
            ("datetime_extractor", FunctionTransformer(extract_datetime_features, validate=False)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("datetime", datetime_transformer, datetime_features))

    # Text features (TF-IDF)
    if text_features:
        text_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("tfidf", TfidfVectorizer(max_features=100, stop_words="english"))
        ])
        transformers.append(("text", text_pipeline, text_features))

    # Final ColumnTransformer
    if not transformers:
        raise ValueError("No valid features found for preprocessing")

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)

    # Fit and transform a small sample
    sample_X = X.sample(n=min(100, len(X)), random_state=42)
    X_transformed = preprocessor.fit_transform(sample_X)

    # Get output feature names
    feature_names = []
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Convert preview to list of dicts
    preview_df = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])
    preview_records = preview_df.head(5).round(4).to_dict(orient="records")

    # Save to session
    session.preprocessor = preprocessor
    session.feature_names_out = feature_names

    summary = {
        "numerical": numerical_features,
        "categorical_low": [c for c in categorical_features if X[c].nunique() <= 10],
        "categorical_high": [c for c in categorical_features if X[c].nunique() > 10],
        "datetime": datetime_features,
        "text": text_features
    }

    return PreprocessingResult(
        pipeline_built=True,
        n_features_in=X.shape[1],
        n_features_out=X_transformed.shape[1],
        transformed_preview=preview_records,
        feature_names_out=feature_names,
        pipeline_summary=summary,
        message="Smart preprocessing pipeline built successfully!"
    )