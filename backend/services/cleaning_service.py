import os
import uuid
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from backend.config import CLEANED_DIR

def clean_dataset(
    df: pd.DataFrame,
    cleaning_steps: List[Dict[str, Any]],
    output_filename: str
) -> Dict[str, Any]:
    """
    Applies an ordered sequence of data cleaning transformations to a dataframe
    and saves the cleaned version without mutating the original.
    """
    df_clean = df.copy()
    logs = []
    initial_shape = df_clean.shape

    for step in cleaning_steps:
        action = step.get("action")
        columns = step.get("columns", [])
        params = step.get("params", {})

        try:
            if action == "drop_duplicates":
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                after = len(df_clean)
                logs.append(f"Removed {before - after} duplicate rows.")

            elif action == "drop_columns":
                valid_cols = [c for c in columns if c in df_clean.columns]
                df_clean = df_clean.drop(columns=valid_cols)
                logs.append(f"Dropped columns: {', '.join(valid_cols)}")

            elif action == "impute_missing":
                strategy = params.get("strategy", "mean") # mean, median, mode, constant, ffill, bfill, drop_rows
                fill_value = params.get("value")
                
                for col in columns:
                    if col not in df_clean.columns:
                        continue
                    null_count = int(df_clean[col].isnull().sum())
                    if null_count == 0:
                        continue

                    if strategy == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        val = df_clean[col].mean()
                        df_clean[col] = df_clean[col].fillna(val)
                        logs.append(f"Imputed {null_count} missing values in '{col}' using mean ({round(val, 2)}).")
                    elif strategy == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(val)
                        logs.append(f"Imputed {null_count} missing values in '{col}' using median ({round(val, 2)}).")
                    elif strategy == "mode":
                        mode_s = df_clean[col].mode()
                        if len(mode_s) > 0:
                            val = mode_s[0]
                            df_clean[col] = df_clean[col].fillna(val)
                            logs.append(f"Imputed {null_count} missing values in '{col}' using mode ('{val}').")
                    elif strategy == "constant" and fill_value is not None:
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        logs.append(f"Imputed {null_count} missing values in '{col}' using constant '{fill_value}'.")
                    elif strategy == "ffill":
                        df_clean[col] = df_clean[col].ffill()
                        logs.append(f"Forward-filled missing values in '{col}'.")
                    elif strategy == "drop_rows":
                        before = len(df_clean)
                        df_clean = df_clean.dropna(subset=[col])
                        after = len(df_clean)
                        logs.append(f"Dropped {before - after} rows containing missing values in '{col}'.")

            elif action == "remove_outliers":
                method = params.get("method", "iqr") # iqr, zscore
                threshold = params.get("threshold", 1.5)

                for col in columns:
                    if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                        continue
                    
                    s = df_clean[col].dropna()
                    if len(s) == 0:
                        continue

                    if method == "iqr":
                        q1 = s.quantile(0.25)
                        q3 = s.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - (threshold * iqr)
                        upper = q3 + (threshold * iqr)
                        before = len(df_clean)
                        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
                        after = len(df_clean)
                        logs.append(f"IQR outlier filtering on '{col}' (bounds [{round(lower, 2)}, {round(upper, 2)}]): removed {before - after} rows.")

                    elif method == "zscore":
                        mean = s.mean()
                        std = s.std()
                        if std > 0:
                            z = (df_clean[col] - mean).abs() / std
                            before = len(df_clean)
                            df_clean = df_clean[z <= threshold]
                            after = len(df_clean)
                            logs.append(f"Z-Score outlier filtering on '{col}' (Z <= {threshold}): removed {before - after} rows.")

            elif action == "rename_column":
                old_name = params.get("old_name")
                new_name = params.get("new_name")
                if old_name in df_clean.columns and new_name:
                    df_clean = df_clean.rename(columns={old_name: new_name})
                    logs.append(f"Renamed column '{old_name}' to '{new_name}'.")

            elif action == "scale_features":
                method = params.get("method", "standard") # standard, minmax
                valid_num_cols = [c for c in columns if c in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[c])]
                if valid_num_cols:
                    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
                    df_clean[valid_num_cols] = scaler.fit_transform(df_clean[valid_num_cols])
                    logs.append(f"Applied {method.title()} Scaling to: {', '.join(valid_num_cols)}")

        except Exception as e:
            logs.append(f"Error in step '{action}': {str(e)}")

    # Save cleaned file
    cleaned_path = CLEANED_DIR / output_filename
    if output_filename.endswith(".xlsx"):
        df_clean.to_excel(cleaned_path, index=False)
    else:
        df_clean.to_csv(cleaned_path, index=False)

    return {
        "initial_shape": initial_shape,
        "final_shape": df_clean.shape,
        "cleaned_path": str(cleaned_path),
        "output_filename": output_filename,
        "logs": logs,
        "row_reduction": initial_shape[0] - df_clean.shape[0]
    }
