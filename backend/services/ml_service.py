import os
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from backend.config import MODELS_DIR

ALGORITHM_MAP = {
    "dummy": ("Dummy Baseline", lambda: DummyRegressor(strategy="mean")),
    "linear_regression": ("Linear Regression", lambda: LinearRegression()),
    "decision_tree": ("Decision Tree Regressor", lambda: DecisionTreeRegressor(random_state=42, max_depth=8, min_samples_leaf=4)),
    "random_forest": ("Random Forest Regressor", lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=12)),
    "xgboost": ("XGBoost Regressor", lambda: xgb.XGBRegressor(n_estimators=120, learning_rate=0.08, max_depth=6, random_state=42, n_jobs=-1))
}

def train_and_evaluate_models(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    selected_algorithms: List[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
    target_unit: str = "quintals/ha"
) -> Dict[str, Any]:
    """
    Trains selected regressors using a clean leakage-free ColumnTransformer pipeline.
    Calculates MAE, RMSE, R2, and K-Fold Cross-Validation scores.
    """
    # 1. Validation & Data Preparation
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset.")
    
    # Filter valid features that exist in df and exclude target column to prevent data leakage
    clean_features = [col for col in feature_columns if col in df.columns and col != target_column]
    if not clean_features:
        raise ValueError("At least one valid feature column must be selected.")

    # Drop rows where target is missing
    df_valid = df.dropna(subset=[target_column]).copy()
    if len(df_valid) < 15:
        raise ValueError(f"Dataset has only {len(df_valid)} valid rows with target; minimum 15 required for ML training.")

    X = df_valid[clean_features]
    y = df_valid[target_column]

    # Categorize features
    numeric_features = [col for col in clean_features if pd.api.types.is_numeric_dtype(X[col])]
    categorical_features = [col for col in clean_features if col not in numeric_features]

    # Preprocessors
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    results = []
    trained_artifacts = {}

    for algo_key in selected_algorithms:
        if algo_key not in ALGORITHM_MAP:
            continue

        algo_name, model_fn = ALGORITHM_MAP[algo_key]
        regressor = model_fn()

        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        start_time = time.time()
        
        # Fit pipeline
        full_pipeline.fit(X_train, y_train)
        training_duration_ms = (time.time() - start_time) * 1000

        # Predict on test set
        y_pred = full_pipeline.predict(X_test)

        # Calculate metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # K-Fold Cross Validation on entire dataset
        try:
            kfold = KFold(n_splits=min(cv_folds, len(df_valid) // 4), shuffle=True, random_state=42)
            cv_scores = cross_val_score(full_pipeline, X, y, cv=kfold, scoring="r2")
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
        except Exception:
            cv_mean = r2
            cv_std = 0.0

        # Feature importances extraction where supported
        feature_importance_dict = {}
        try:
            # Extract feature names after one-hot encoding
            fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
            feature_names = []
            if numeric_features:
                feature_names.extend(numeric_features)
            if categorical_features:
                cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
                encoded_cats = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(list(encoded_cats))

            fitted_model = full_pipeline.named_steps["model"]
            if hasattr(fitted_model, "feature_importances_"):
                importances = fitted_model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1][:15]
                for idx in sorted_idx:
                    if idx < len(feature_names):
                        feature_importance_dict[str(feature_names[idx])] = round(float(importances[idx]), 4)
            elif hasattr(fitted_model, "coef_"):
                coefs = np.abs(fitted_model.coef_)
                sorted_idx = np.argsort(coefs)[::-1][:15]
                for idx in sorted_idx:
                    if idx < len(feature_names):
                        feature_importance_dict[str(feature_names[idx])] = round(float(coefs[idx]), 4)
        except Exception:
            pass

        # Actual vs Predicted sample pairs (first 30 points)
        sample_size = min(30, len(y_test))
        actual_vs_pred = [
            {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
            for a, p in zip(list(y_test)[:sample_size], list(y_pred)[:sample_size])
        ]

        # Save pipeline artifact to disk
        model_id = str(uuid.uuid4())
        model_filename = f"model_{algo_key}_{model_id[:8]}.joblib"
        model_path = MODELS_DIR / model_filename

        metadata = {
            "model_id": model_id,
            "algorithm": algo_key,
            "algorithm_name": algo_name,
            "target_column": target_column,
            "target_unit": target_unit,
            "feature_columns": clean_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "cv_mean_r2": round(cv_mean, 4),
                "cv_std_r2": round(cv_std, 4),
                "duration_ms": round(training_duration_ms, 2)
            }
        }

        joblib.dump({"pipeline": full_pipeline, "metadata": metadata}, model_path)

        results.append({
            "model_id": model_id,
            "algorithm": algo_key,
            "name": algo_name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "cv_mean_r2": round(cv_mean, 4),
            "cv_std_r2": round(cv_std, 4),
            "training_duration_ms": round(training_duration_ms, 2),
            "model_path": str(model_path),
            "feature_importances": feature_importance_dict,
            "actual_vs_predicted": actual_vs_pred,
            "metadata": metadata
        })

    # Sort results by R2 score descending to identify best model
    results.sort(key=lambda x: x["r2"], reverse=True)
    if results:
        results[0]["is_best"] = True

    return {
        "target_column": target_column,
        "target_unit": target_unit,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(clean_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": results,
        "best_model_id": results[0]["model_id"] if results else None
    }

def load_trained_pipeline(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model artifact not found at {model_path}")
    
    data = joblib.load(path)
    return data["pipeline"], data["metadata"]
