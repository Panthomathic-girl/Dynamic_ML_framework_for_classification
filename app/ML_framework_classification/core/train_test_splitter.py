# app/ML_framework_classification/core/splitter/train_test_splitter.py
from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
import numpy as np
from app.ML_framework_classification.core.session import session

def perform_stratified_split(
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True
) -> dict:
    """
    Performs stratified train/test split and saves to session.
    """
    if session.preprocessor is None:
        raise ValueError("Preprocessor not built. Call /build-preprocessor/ first.")
    if session.X is None or session.y is None:
        raise ValueError("Features/target not selected.")

    X = session.X
    y = session.y

    # Apply full preprocessing
    X_processed = session.preprocessor.fit_transform(X)
    y_array = y.values if isinstance(y, pd.Series) else y

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_array,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y_array
    )

    # Save to session
    session.X_train = X_train
    session.X_test = X_test
    session.y_train = y_train
    session.y_test = y_test
    session.X_processed = X_processed
    session.y_processed = y_array
    session.cv_strategy = "train_test"
    session.split_config = {
        "method": "stratified_train_test",
        "test_size": test_size,
        "random_state": random_state
    }

    return {
        "method": "stratified_train_test",
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "test_size": test_size,
        "message": f"Stratified split completed: {X_train.shape[0]} train, {X_test.shape[0]} test",
        "ready_for_training": True
    }