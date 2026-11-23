import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.config import UPLOAD_DIR, DATA_DIR

def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Attempt auto-detection of delimiter
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep=None, engine='python')
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif suffix == ".json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    row_count, col_count = df.shape
    columns_info = []
    numeric_cols = []
    categorical_cols = []
    date_cols = []

    for col in df.columns:
        col_series = df[col]
        dtype_str = str(col_series.dtype)
        null_count = int(col_series.isnull().sum())
        null_pct = round((null_count / max(row_count, 1)) * 100, 2)
        unique_count = int(col_series.nunique())

        col_type = "categorical"
        if pd.api.types.is_numeric_dtype(col_series):
            col_type = "numeric"
            numeric_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(col_series):
            col_type = "datetime"
            date_cols.append(col)
        else:
            # Check if strings might be parseable dates
            sample_non_null = col_series.dropna().head(10)
            is_date = False
            if len(sample_non_null) > 0:
                try:
                    pd.to_datetime(sample_non_null, errors='raise')
                    is_date = True
                except Exception:
                    is_date = False
            
            if is_date:
                col_type = "datetime"
                date_cols.append(col)
            else:
                categorical_cols.append(col)

        col_stat: Dict[str, Any] = {
            "name": str(col),
            "type": col_type,
            "raw_dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
        }

        if col_type == "numeric":
            clean_s = col_series.dropna()
            if len(clean_s) > 0:
                col_stat.update({
                    "min": float(clean_s.min()),
                    "max": float(clean_s.max()),
                    "mean": round(float(clean_s.mean()), 2),
                    "median": round(float(clean_s.median()), 2),
                    "std": round(float(clean_s.std()), 2) if len(clean_s) > 1 else 0.0,
                    "skew": round(float(clean_s.skew()), 2) if len(clean_s) > 2 else 0.0,
                })
        else:
            top_vals = col_series.value_counts().head(5).to_dict()
            col_stat["top_values"] = {str(k): int(v) for k, v in top_vals.items()}

        columns_info.append(col_stat)

    # Missing value matrix / summary
    missing_total = int(df.isnull().sum().sum())
    duplicates_count = int(df.duplicated().sum())

    # Correlation matrix for numeric columns (limit to top 15 numeric columns for performance)
    correlation_matrix: Dict[str, Dict[str, float]] = {}
    if len(numeric_cols) > 1:
        corr_subset = numeric_cols[:15]
        corr_df = df[corr_subset].corr().fillna(0)
        for c1 in corr_subset:
            correlation_matrix[c1] = {}
            for c2 in corr_subset:
                val = corr_df.loc[c1, c2]
                correlation_matrix[c1][c2] = round(float(val), 3) if not np.isnan(val) else 0.0

    # Target recommendations
    recommended_targets = []
    target_keywords = ["yield", "production", "output", "harvest", "crop_yield", "amount", "sales", "revenue"]
    for col in numeric_cols:
        col_lower = str(col).lower()
        priority = 0
        for kw in target_keywords:
            if kw in col_lower:
                priority += 10
        if priority > 0:
            recommended_targets.append({"column": col, "confidence": "High", "reason": f"Matches standard agricultural target keywords ({col})"})
        elif df[col].nunique() > 10:
            recommended_targets.append({"column": col, "confidence": "Medium", "reason": "Continuous numeric variable with good variance"})

    # Sort recommendations
    recommended_targets.sort(key=lambda x: (x["confidence"] != "High", x["column"]))

    # Overall dataset quality score (0 - 100)
    # Deductions: null percentage, duplicate percentage, small sample size
    null_ratio = missing_total / max(row_count * col_count, 1)
    dup_ratio = duplicates_count / max(row_count, 1)
    quality_score = max(10, int(100 - (null_ratio * 150) - (dup_ratio * 100)))

    return {
        "row_count": row_count,
        "col_count": col_count,
        "columns": columns_info,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": date_cols,
        "missing_total": missing_total,
        "missing_pct": round((missing_total / max(row_count * col_count, 1)) * 100, 2),
        "duplicates_count": duplicates_count,
        "quality_score": quality_score,
        "correlation_matrix": correlation_matrix,
        "recommended_targets": recommended_targets,
    }

def get_preview(df: pd.DataFrame, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    total = len(df)
    subset = df.iloc[offset : offset + limit]
    # Replace NaN with None for clean JSON serialization
    records = subset.replace({np.nan: None}).to_dict(orient="records")
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "columns": [str(c) for c in df.columns],
        "rows": records
    }
