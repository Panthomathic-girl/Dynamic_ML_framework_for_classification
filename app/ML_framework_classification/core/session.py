# app/ML_framework_classification/core/session.py
from typing import Optional
import pandas as pd

class SessionState:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.X_cols: Optional[list] = None
        self.y_col: Optional[str] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.file_path: Optional[str] = None

# Single in-memory session (for demo). Use Redis in production.
session = SessionState()