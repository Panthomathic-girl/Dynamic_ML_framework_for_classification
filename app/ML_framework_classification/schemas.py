# app/ML_framework_classification/schemas.py
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UploadResponse(BaseModel):
    columns: List[str]
    shape: tuple
    preview: List[Dict[str, Any]]
    message: str = "CSV uploaded successfully"

class ColumnSelectionRequest(BaseModel):
    X_cols: List[str]
    y_col: str

class ColumnSelectionResponse(BaseModel):
    message: str
    X_cols: List[str]
    y_col: str
    X_shape: tuple
    y_shape: tuple
    target_classes: List[str]
    class_distribution: Dict[str, int]
    
class FeatureInfo(BaseModel):
    name: str
    inferred_type: str = Field(..., description="numerical, categorical, boolean, datetime, text")
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    cardinality_pct: float = Field(..., description="unique / total rows")
    sample_values: List[Any] = Field(..., description="First 5 non-null values")
    is_target: bool = False

class TargetAnalysis(BaseModel):
    classes: List[str]
    distribution: Dict[str, int]
    n_classes: int
    imbalance_ratio: Optional[float] = None  # max/min
    is_imbalanced: bool = False
    recommendation: Optional[str] = None

class DataUnderstandingReport(BaseModel):
    total_rows: int
    total_columns: int
    features: List[FeatureInfo]
    target: TargetAnalysis
    quality_flags: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    generated_at: str
    
class ValidationErrorDetail(BaseModel):
    check: str
    passed: bool
    message: str
    severity: str  # "error" or "warning"

class DataValidationReport(BaseModel):
    overall_passed: bool
    errors: List[ValidationErrorDetail] = Field(default_factory=list)
    warnings: List[ValidationErrorDetail] = Field(default_factory=list)
    summary: str
    
class PreprocessingResult(BaseModel):
    pipeline_built: bool
    n_features_in: int
    n_features_out: int
    transformed_preview: List[Dict[str, Any]]
    feature_names_out: List[str]
    pipeline_summary: Dict[str, List[str]]
    message: str
    
class SplitMethod(str, Enum):
    train_test = "train_test"
    kfold = "kfold"

class TrainTestSplitRequest(BaseModel):
    test_size: float = Field(0.2, ge=0.1, le=0.5)
    random_state: int = Field(42, ge=0)
    shuffle: bool = True

class KFoldRequest(BaseModel):
    n_splits: int = Field(5, ge=3, le=10)
    random_state: int = Field(42, ge=0)
    shuffle: bool = True

class SplitResponse(BaseModel):
    method: str
    train_shape: Optional[tuple] = None
    test_shape: Optional[tuple] = None
    n_splits: Optional[int] = None
    fold_summary: Optional[List[dict]] = None
    message: str
    ready_for_training: bool
    
    
# class FinalTrainingReport(BaseModel):
#     best_model: str
#     best_cv_f1_macro: float
#     cv_mean_f1_macro: float
#     cv_std_f1_macro: float
#     total_models_tested: int
#     all_results: List[Dict]
#     model_file: str
#     test_performance: Dict
#     timestamp: str
#     message: str 

class FinalTrainingReport(BaseModel):
    best_model: str
    best_cv_f1_macro: float
    cv_mean_f1_macro: float
    cv_std_f1_macro: float
    total_models_tested: int
    all_results: List[Dict[str, Any]]        # Only summary per model
    model_file: str
    test_performance: Dict[str, Any]         # Full metrics for best model only
    timestamp: str
    message: str   
    

# class ModelResult(BaseModel):
#     model: str
#     cv_f1_macro: float
#     best_params: Dict[str, Any]
#     model_file: str
#     test_performance: Dict[str, Any]  # Full metrics for this model

# class FinalTrainingReport(BaseModel):
#     best_model: str
#     best_cv_f1_macro: float
#     cv_mean_f1_macro: float
#     cv_std_f1_macro: float
#     total_models_tested: int
#     all_results: List[ModelResult]        # Now rich with full metrics
#     model_file: str
#     test_performance: Dict[str, Any]
#     timestamp: str
#     message: str

    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }    
   
        