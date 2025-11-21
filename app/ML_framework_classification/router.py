# app/ML_framework_classification/router.py

import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List

# Core functions
from .core.upload_handler import handle_csv_upload
from .core.column_selector import select_columns
from .core.data_analyzer import analyze_dataset
from .core.data_validator import validate_dataset
from .core.preprocessor import build_preprocessing_pipeline
from .core.train_test_splitter import perform_stratified_split
from .core.kfold_splitter import setup_stratified_kfold
from .core.trainer_v2 import run_full_automl
from .core import session

# Response model
from .schemas import FinalTrainingReport

router = APIRouter()


# =============================================
# SINGLE MASTER API — ONE CALL = FULL AUTOML
# =============================================
@router.post("/run-automl-full/", response_model=FinalTrainingReport)
async def run_automl_full(
    file: UploadFile = File(...),
    X_cols: str = Form(...),           # JSON string like '["col1","col2"]'
    y_col: str = Form(...),
    test_size: float = Form(0.2),      # Default: 80/20 split
    use_kfold: bool = Form(False),     # False = train/test, True = KFold
    n_splits: int = Form(5),           # Only used if use_kfold=True
):
    """
    ONE API CALL → FULL END-TO-END CLASSIFICATION AUTOML

    Upload CSV + select columns → AutoML runs immediately
    Returns best model + full metrics + saved .pkl file
    """

    # Step 1: Upload CSV
    df = await handle_csv_upload(file)

    # Step 2: Parse X_cols from JSON string
    try:
        X_cols_list: List[str] = json.loads(X_cols)
        if not isinstance(X_cols_list, list) or not all(isinstance(c, str) for c in X_cols_list):
            raise ValueError("X_cols must be a list of strings")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X_cols format. Use JSON array string, e.g. '[\"col1\",\"col2\"]'. Error: {e}"
        )

    # Step 3: Validate columns exist in dataframe
    missing_cols = [col for col in X_cols_list + [y_col] if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in CSV: {missing_cols}"
        )

    # Step 4: Select features and target
    select_columns(X_cols_list, y_col)

    # Step 5: Run analysis & validation (silent — just to initialize)
    try:
        analyze_dataset()
        validate_dataset()
    except Exception as e:
        print(f"Warning during analysis/validation: {e}")

    # Step 6: Build preprocessing pipeline
    try:
        build_preprocessing_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build preprocessor: {e}")

    # Step 7: Data splitting
    try:
        if use_kfold:
            setup_stratified_kfold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            perform_stratified_split(test_size=test_size, random_state=42, shuffle=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data split failed: {e}")

    # Step 8: RUN FULL AUTOML → BEST MODEL
    try:
        result = run_full_automl()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AutoML training failed: {str(e)}")