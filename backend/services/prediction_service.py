import uuid
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from backend.services.ml_service import load_trained_pipeline

def predict_single(
    model_path: str,
    inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates input features, runs inference through the trained pipeline,
    and returns predicted yield with confidence interval and metadata.
    """
    pipeline, metadata = load_trained_pipeline(model_path)
    required_features = metadata.get("feature_columns", [])
    target_unit = metadata.get("target_unit", "quintals/ha")
    metrics = metadata.get("metrics", {})

    # Validate that required features are present
    missing_features = [f for f in required_features if f not in inputs]
    if missing_features:
        raise ValueError(f"Missing required input features for this model: {', '.join(missing_features)}")

    # Prepare single-row DataFrame
    input_data = {f: [inputs.get(f)] for f in required_features}
    df_input = pd.DataFrame(input_data)

    # Cast numeric types if possible
    numeric_features = metadata.get("numeric_features", [])
    for col in numeric_features:
        if col in df_input:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

    # Predict
    raw_pred = pipeline.predict(df_input)[0]
    predicted_val = max(0.0, float(raw_pred))

    # Calculate uncertainty based on model test RMSE
    rmse = metrics.get("rmse", 2.0)
    uncertainty_margin = float(rmse * 1.96) # 95% approximate prediction bound
    conf_lower = max(0.0, predicted_val - uncertainty_margin)
    conf_upper = predicted_val + uncertainty_margin
    uncertainty_pct = round((uncertainty_margin / max(predicted_val, 1.0)) * 100, 1)

    prediction_id = str(uuid.uuid4())

    return {
        "prediction_id": prediction_id,
        "predicted_yield": round(predicted_val, 2),
        "target_unit": target_unit,
        "confidence_lower": round(conf_lower, 2),
        "confidence_upper": round(conf_upper, 2),
        "uncertainty_pct": min(uncertainty_pct, 100.0),
        "model_name": metadata.get("algorithm_name", "Trained Model"),
        "algorithm": metadata.get("algorithm", "regressor"),
        "metrics": metrics,
        "inputs": inputs,
    }
