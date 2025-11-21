# app/ML_framework_classification/core/column_selector.py
from fastapi import HTTPException
from .session import session

def select_columns(X_cols: list, y_col: str):
    if session.df is None:
        raise HTTPException(status_code=400, detail="No CSV uploaded")

    df = session.df

    all_selected = X_cols + [y_col]
    missing = [c for c in all_selected if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found: {missing}")

    if y_col in X_cols:
        raise HTTPException(status_code=400, detail="Target cannot be in features")

    if len(X_cols) == 0:
        raise HTTPException(status_code=400, detail="Select at least one feature")

    X = df[X_cols]
    y = df[y_col]

    # Update session
    session.X_cols = X_cols
    session.y_col = y_col
    session.X = X
    session.y = y

    # FIXED: Convert keys to string to satisfy Pydantic Dict[str, int]
    class_dist = y.value_counts()
    class_distribution = {str(k): int(v) for k, v in class_dist.to_dict().items()}

    return {
        "X_shape": X.shape,
        "y_shape": y.shape,
        "target_classes": [str(c) for c in y.dropna().unique().tolist()],
        "class_distribution": class_distribution
    }