import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.config import EXPORTS_DIR
from backend.services.ml_service import load_trained_pipeline
from backend.services.data_service import load_dataframe

def suggest_column_mapping(file_columns: List[str], model_features: List[str]) -> Dict[str, Optional[str]]:
    """
    Automatically suggests a mapping from model required feature -> dataset column.
    """
    mapping = {}
    normalized_file_cols = {c.lower().replace("_", "").replace(" ", "").replace("-", ""): c for c in file_columns}

    for feature in model_features:
        feat_norm = feature.lower().replace("_", "").replace(" ", "").replace("-", "")
        if feat_norm in normalized_file_cols:
            mapping[feature] = normalized_file_cols[feat_norm]
        else:
            # Check for substring match
            found = None
            for f_norm, original in normalized_file_cols.items():
                if f_norm in feat_norm or feat_norm in f_norm:
                    found = original
                    break
            mapping[feature] = found

    return mapping

def run_batch_prediction(
    file_path: str,
    model_path: str,
    column_mapping: Dict[str, str], # model_feature -> file_column
    actual_target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads batch dataset, maps columns to model input requirements,
    runs vectorized pipeline inference, computes distribution stats and optional evaluation metrics,
    and writes out enriched CSV and Excel export files.
    """
    df = load_dataframe(file_path)
    pipeline, metadata = load_trained_pipeline(model_path)
    required_features = metadata.get("feature_columns", [])
    target_unit = metadata.get("target_unit", "quintals/ha")
    numeric_features = metadata.get("numeric_features", [])

    # Validate mapped columns exist
    missing_mappings = [feat for feat in required_features if feat not in column_mapping or not column_mapping[feat]]
    if missing_mappings:
        raise ValueError(f"Mapping missing for required features: {', '.join(missing_mappings)}")

    for feat, col in column_mapping.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' mapped to '{feat}' does not exist in uploaded file.")

    # Construct input dataframe matching exact feature names
    df_features = pd.DataFrame()
    for feat in required_features:
        source_col = column_mapping[feat]
        df_features[feat] = df[source_col]
        if feat in numeric_features:
            df_features[feat] = pd.to_numeric(df_features[feat], errors="coerce")

    # Generate predictions
    predictions = pipeline.predict(df_features)
    predictions = np.maximum(0.0, predictions) # Yield cannot be negative

    # Append prediction column to original dataframe
    result_df = df.copy()
    pred_col_name = f"Predicted_Yield_{target_unit.replace('/', '_')}"
    result_df[pred_col_name] = [round(float(p), 2) for p in predictions]

    # Optional evaluation if actual yield column is supplied
    evaluation = None
    if actual_target_column and actual_target_column in df.columns:
        # Check target wasn't used as an input feature
        if actual_target_column not in column_mapping.values():
            y_actual = pd.to_numeric(df[actual_target_column], errors="coerce")
            valid_mask = ~y_actual.isna()
            if valid_mask.sum() > 2:
                y_act_clean = y_actual[valid_mask]
                y_pred_clean = predictions[valid_mask]
                mae = float(mean_absolute_error(y_act_clean, y_pred_clean))
                rmse = float(np.sqrt(mean_squared_error(y_act_clean, y_pred_clean)))
                r2 = float(r2_score(y_act_clean, y_pred_clean))
                evaluation = {
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                    "evaluated_records": int(valid_mask.sum())
                }

    # Generate distribution metrics
    stats = {
        "count": len(predictions),
        "mean": round(float(np.mean(predictions)), 2),
        "median": round(float(np.median(predictions)), 2),
        "std": round(float(np.std(predictions)), 2),
        "min": round(float(np.min(predictions)), 2),
        "max": round(float(np.max(predictions)), 2),
    }

    # Generate histogram bins for UI visualization
    hist, bin_edges = np.histogram(predictions, bins=10)
    distribution_bins = [
        {"bin": f"{round(bin_edges[i], 1)} - {round(bin_edges[i+1], 1)}", "count": int(hist[i])}
        for i in range(len(hist))
    ]

    # Save enriched export file
    batch_id = str(uuid.uuid4())[:8]
    export_csv_filename = f"batch_predictions_{batch_id}.csv"
    export_xlsx_filename = f"batch_predictions_{batch_id}.xlsx"
    export_csv_path = EXPORTS_DIR / export_csv_filename
    export_xlsx_path = EXPORTS_DIR / export_xlsx_filename

    result_df.to_csv(export_csv_path, index=False)
    result_df.to_excel(export_xlsx_path, index=False)

    # Preview rows (up to 50)
    preview_subset = result_df.head(50).replace({np.nan: None}).to_dict(orient="records")

    return {
        "batch_id": batch_id,
        "total_records": len(predictions),
        "prediction_column": pred_col_name,
        "target_unit": target_unit,
        "stats": stats,
        "distribution_bins": distribution_bins,
        "evaluation": evaluation,
        "export_csv": export_csv_filename,
        "export_xlsx": export_xlsx_filename,
        "preview_columns": list(result_df.columns),
        "preview_records": preview_subset
    }
