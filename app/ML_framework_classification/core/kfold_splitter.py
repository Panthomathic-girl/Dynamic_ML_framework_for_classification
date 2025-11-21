# app/ML_framework_classification/core/splitter/kfold_splitter.py
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from app.ML_framework_classification.core.session import session

def setup_stratified_kfold(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> dict:
    """
    Sets up StratifiedKFold and saves to session.
    """
    if session.preprocessor is None:
        raise ValueError("Preprocessor not built. Call /build-preprocessor/ first.")
    if session.X is None or session.y is None:
        raise ValueError("Features/target not selected.")

    X = session.X
    y = session.y

    X_processed = session.preprocessor.fit_transform(X)
    y_array = y.values if isinstance(y, pd.Series) else y

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    # Save everything
    session.X_processed = X_processed
    session.y_processed = y_array
    session.kfold = skf
    session.n_splits = n_splits
    session.cv_strategy = "kfold"
    session.split_config = {
        "method": "stratified_kfold",
        "n_splits": n_splits,
        "random_state": random_state
    }

    # Generate fold summary
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_array)):
        folds.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx)
        })

    return {
        "method": "stratified_kfold",
        "n_splits": n_splits,
        "total_samples": X_processed.shape[0],
        "fold_summary": folds,
        "message": f"Stratified {n_splits}-Fold CV setup complete. Ready for cross-validated training.",
        "ready_for_training": True
    }

def get_kfold_plan() -> dict:
    """Returns detailed fold distribution (class balance per fold)"""
    if session.cv_strategy != "kfold" or session.kfold is None:
        raise ValueError("K-Fold not set up. Call setup_stratified_kfold() first.")

    X_proc = session.X_processed
    y_proc = session.y_processed
    folds_detail = []

    for fold_idx, (train_idx, val_idx) in enumerate(session.kfold.split(X_proc, y_proc)):
        train_dist = dict(pd.Series(y_proc[train_idx]).value_counts())
        val_dist = dict(pd.Series(y_proc[val_idx]).value_counts())

        folds_detail.append({
            "fold": fold_idx + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_class_distribution": train_dist,
            "val_class_distribution": val_dist
        })

    return {
        "cv_strategy": "stratified_kfold",
        "n_splits": session.n_splits,
        "folds": folds_detail
    }