# app/ML_framework_classification/core/upload_handler.py
from fastapi import UploadFile, HTTPException
import pandas as pd
import tempfile
import os
from .session import session

async def handle_csv_upload(file: UploadFile):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        df = pd.read_csv(temp_path)
    except Exception as e:
        os.unlink(temp_path)
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

    if df.empty or len(df.columns) < 2:
        os.unlink(temp_path)
        raise HTTPException(status_code=400, detail="CSV must have ≥2 columns and not be empty")

    # Update session
    session.df = df.copy()
    session.file_path = temp_path
    session.X_cols = None
    session.y_col = None

    return df