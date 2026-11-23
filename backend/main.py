import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import UPLOAD_DIR, CLEANED_DIR, EXPORTS_DIR, MODELS_DIR
from backend.database import get_db, init_db, DatasetModel, MLModel, PredictionLog, BatchPredictionLog
from backend.services.data_service import load_dataframe, profile_dataset, get_preview
from backend.services.cleaning_service import clean_dataset
from backend.services.ml_service import train_and_evaluate_models, load_trained_pipeline
from backend.services.prediction_service import predict_single
from backend.services.batch_service import suggest_column_mapping, run_batch_prediction
from backend.services.agrimall_service import parse_agrimall_data
from backend.services.forecast_service import generate_demand_forecast
from backend.services.report_service import generate_pdf_report
from backend.seed_data import seed_database_if_needed

app = FastAPI(
    title="AgriPredict AI API",
    description="Agricultural Intelligence Platform API for Crop Yield Prediction and Agri Mall Analytics",
    version="2.4.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    seed_database_if_needed()

# ==================== Health & System ====================
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AgriPredict AI Engine",
        "version": "2.4.0",
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== Overview / KPI ====================
@app.get("/api/overview")
def get_overview_kpis(db: Session = Depends(get_db)):
    total_datasets = db.query(DatasetModel).count()
    active_dataset = db.query(DatasetModel).filter(DatasetModel.is_active == True).first()
    
    total_records = 0
    all_datasets = db.query(DatasetModel).all()
    for ds in all_datasets:
        total_records += (ds.row_count or 0)

    total_models = db.query(MLModel).count()
    best_model = db.query(MLModel).filter(MLModel.is_best == True).first()
    latest_prediction = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).first()

    crop_count = 10
    if active_dataset and active_dataset.meta_json:
        meta = json.loads(active_dataset.meta_json)
        for col_info in meta.get("columns", []):
            if col_info["name"].lower() in ["crop", "crop_name", "croptype"]:
                crop_count = col_info.get("unique_count", 10)

    # Ingest Agri Mall stats
    mall_summary = parse_agrimall_data()

    return {
        "kpis": {
            "total_datasets": total_datasets,
            "total_records": total_records,
            "crop_count": crop_count,
            "trained_models": total_models,
            "best_model_name": best_model.name if best_model else "None",
            "best_model_r2": round(best_model.r2, 4) if best_model and best_model.r2 else 0.0,
            "latest_prediction": latest_prediction.to_dict() if latest_prediction else None,
            "dataset_quality_score": json.loads(active_dataset.meta_json).get("quality_score", 95) if active_dataset and active_dataset.meta_json else 95,
            "agrimall_revenue": mall_summary.get("total_revenue", 0.0),
            "agrimall_units_sold": mall_summary.get("total_quantity_sold", 0)
        },
        "active_dataset": active_dataset.to_dict() if active_dataset else None
    }

# ==================== Datasets Endpoints ====================
@app.get("/api/datasets")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(DatasetModel).order_by(DatasetModel.created_at.desc()).all()
    return [d.to_dict() for d in datasets]

@app.post("/api/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".csv", ".xlsx", ".xls", ".json"]:
        raise HTTPException(status_code=400, detail="Unsupported format. Only CSV, Excel (.xlsx, .xls), and JSON are allowed.")

    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id[:8]}_{file.filename}"
    save_path = UPLOAD_DIR / stored_filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = load_dataframe(str(save_path))
        profile = profile_dataset(df)
    except Exception as e:
        if save_path.exists():
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Failed to read/profile file: {str(e)}")

    display_name = name or Path(file.filename).stem

    dataset = DatasetModel(
        id=file_id,
        name=display_name,
        filename=stored_filename,
        file_path=str(save_path),
        file_type=suffix.replace(".", ""),
        row_count=len(df),
        col_count=len(df.columns),
        created_at=datetime.utcnow(),
        is_active=False,
        is_cleaned=False,
        meta_json=json.dumps(profile)
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset.to_dict()

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset.to_dict()

@app.get("/api/datasets/{dataset_id}/preview")
def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataframe(dataset.file_path)

    if search:
        search_lower = search.lower()
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(search_lower).any(), axis=1)
        df = df[mask]

    return get_preview(df, limit=limit, offset=offset)

@app.put("/api/datasets/{dataset_id}/rename")
def rename_dataset(dataset_id: str, payload: Dict[str, str], db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    new_name = payload.get("name", "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")
    dataset.name = new_name
    db.commit()
    return dataset.to_dict()

@app.post("/api/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Deactivate all others
    db.query(DatasetModel).update({DatasetModel.is_active: False})
    dataset.is_active = True
    db.commit()
    return dataset.to_dict()

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Remove file if safe
    try:
        p = Path(dataset.file_path)
        if p.exists() and "uploads" in str(p):
            p.unlink()
    except Exception:
        pass

    db.delete(dataset)
    db.commit()
    return {"message": "Dataset deleted successfully"}

@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == dataset_id).first()
    if not dataset or not Path(dataset.file_path).exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    return FileResponse(dataset.file_path, filename=f"{dataset.name}.{dataset.file_type}")

@app.get("/api/datasets/compare/{id1}/{id2}")
def compare_datasets(id1: str, id2: str, db: Session = Depends(get_db)):
    ds1 = db.query(DatasetModel).filter(DatasetModel.id == id1).first()
    ds2 = db.query(DatasetModel).filter(DatasetModel.id == id2).first()
    if not ds1 or not ds2:
        raise HTTPException(status_code=404, detail="One or both datasets not found")

    m1 = json.loads(ds1.meta_json) if ds1.meta_json else {}
    m2 = json.loads(ds2.meta_json) if ds2.meta_json else {}

    cols1 = set(c["name"] for c in m1.get("columns", []))
    cols2 = set(c["name"] for c in m2.get("columns", []))

    return {
        "dataset1": ds1.to_dict(),
        "dataset2": ds2.to_dict(),
        "common_columns": list(cols1.intersection(cols2)),
        "unique_to_1": list(cols1 - cols2),
        "unique_to_2": list(cols2 - cols1),
        "row_difference": ds1.row_count - ds2.row_count,
        "quality_score_diff": m1.get("quality_score", 0) - m2.get("quality_score", 0)
    }

# ==================== Data Cleaning ====================
class CleanDatasetRequest(BaseModel):
    dataset_id: str
    cleaned_name: str
    steps: List[Dict[str, Any]]

@app.post("/api/cleaning/clean")
def clean_dataset_endpoint(payload: CleanDatasetRequest, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataframe(dataset.file_path)
    output_filename = f"cleaned_{uuid.uuid4().hex[:8]}_{dataset.filename}"
    
    result = clean_dataset(
        df=df,
        cleaning_steps=payload.steps,
        output_filename=output_filename
    )

    # Profile the new cleaned dataframe
    df_cleaned = load_dataframe(result["cleaned_path"])
    profile = profile_dataset(df_cleaned)

    # Save as new dataset record
    new_id = str(uuid.uuid4())
    new_ds = DatasetModel(
        id=new_id,
        name=payload.cleaned_name or f"Cleaned - {dataset.name}",
        filename=output_filename,
        file_path=result["cleaned_path"],
        file_type="csv",
        row_count=len(df_cleaned),
        col_count=len(df_cleaned.columns),
        created_at=datetime.utcnow(),
        is_active=False,
        is_cleaned=True,
        parent_id=dataset.id,
        meta_json=json.dumps(profile)
    )
    db.add(new_ds)
    db.commit()

    return {
        "new_dataset": new_ds.to_dict(),
        "cleaning_summary": result
    }

# ==================== AI Data Explorer ====================
class ChartDataRequest(BaseModel):
    dataset_id: str
    chart_type: str # histogram, bar, line, scatter, boxplot, heatmap
    x_axis: str
    y_axis: Optional[str] = None
    group_by: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

@app.post("/api/explorer/chart-data")
def get_chart_data(payload: ChartDataRequest, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataframe(dataset.file_path)
    
    # Apply filters if provided
    if payload.filters:
        for col, val in payload.filters.items():
            if col in df.columns and val is not None and val != "":
                df = df[df[col] == val]

    if payload.x_axis not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{payload.x_axis}' not found in dataset")

    chart_type = payload.chart_type.lower()
    chart_payload: Dict[str, Any] = {"type": chart_type, "x_axis": payload.x_axis, "data": []}

    if chart_type == "histogram":
        s = pd.to_numeric(df[payload.x_axis], errors="coerce").dropna()
        hist, bin_edges = np.histogram(s, bins=12)
        chart_payload["data"] = [
            {"bin": f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}", "count": int(hist[i])}
            for i in range(len(hist))
        ]
        chart_payload["stats"] = {
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "std": round(float(s.std()), 2) if len(s) > 1 else 0.0
        }

    elif chart_type == "bar":
        if payload.y_axis and payload.y_axis in df.columns:
            y_num = pd.to_numeric(df[payload.y_axis], errors="coerce")
            grp = df.assign(_y=y_num).groupby(payload.x_axis)["_y"].mean().reset_index().dropna()
            grp = grp.sort_values(by="_y", ascending=False).head(20)
            chart_payload["data"] = [{"label": str(r[payload.x_axis]), "value": round(float(r["_y"]), 2)} for _, r in grp.iterrows()]
        else:
            vc = df[payload.x_axis].value_counts().head(20)
            chart_payload["data"] = [{"label": str(k), "value": int(v)} for k, v in vc.items()]

    elif chart_type in ["scatter", "line"]:
        if not payload.y_axis or payload.y_axis not in df.columns:
            raise HTTPException(status_code=400, detail="Y-Axis is required for scatter/line charts")
        sub = df[[payload.x_axis, payload.y_axis]].dropna().head(100)
        chart_payload["data"] = [
            {"x": float(r[payload.x_axis]) if pd.api.types.is_numeric_dtype(df[payload.x_axis]) else str(r[payload.x_axis]),
             "y": round(float(r[payload.y_axis]), 2)}
            for _, r in sub.iterrows()
        ]

    elif chart_type == "boxplot":
        if payload.y_axis and payload.y_axis in df.columns:
            y_s = pd.to_numeric(df[payload.y_axis], errors="coerce").dropna()
            groups = df.groupby(payload.x_axis)
            box_data = []
            for g_name, g_df in groups:
                vals = pd.to_numeric(g_df[payload.y_axis], errors="coerce").dropna()
                if len(vals) > 0:
                    box_data.append({
                        "group": str(g_name),
                        "min": round(float(vals.min()), 2),
                        "q1": round(float(vals.quantile(0.25)), 2),
                        "median": round(float(vals.median()), 2),
                        "q3": round(float(vals.quantile(0.75)), 2),
                        "max": round(float(vals.max()), 2)
                    })
            chart_payload["data"] = box_data[:15]

    return chart_payload

# ==================== ML Model Studio ====================
@app.get("/api/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(MLModel).order_by(MLModel.r2.desc()).all()
    return [m.to_dict() for m in models]

@app.get("/api/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.to_dict()

class TrainModelsRequest(BaseModel):
    dataset_id: str
    target_column: str
    feature_columns: List[str]
    selected_algorithms: List[str]
    test_size: float = 0.2
    target_unit: str = "quintals/ha"

@app.post("/api/models/train")
def train_models_endpoint(payload: TrainModelsRequest, db: Session = Depends(get_db)):
    dataset = db.query(DatasetModel).filter(DatasetModel.id == payload.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = load_dataframe(dataset.file_path)

    try:
        results = train_and_evaluate_models(
            df=df,
            target_column=payload.target_column,
            feature_columns=payload.feature_columns,
            selected_algorithms=payload.selected_algorithms,
            test_size=payload.test_size,
            target_unit=payload.target_unit
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Clear previous 'best' flag if new best is trained
    db.query(MLModel).update({MLModel.is_best: False})

    saved_records = []
    best_id = results.get("best_model_id")

    for m in results.get("models", []):
        is_b = (m["model_id"] == best_id)
        ml_record = MLModel(
            id=m["model_id"],
            name=f"{m['name']} ({payload.target_column})",
            algorithm=m["algorithm"],
            dataset_id=dataset.id,
            target_column=payload.target_column,
            target_unit=payload.target_unit,
            feature_columns=json.dumps(payload.feature_columns),
            categorical_features=json.dumps(results.get("categorical_features", [])),
            numerical_features=json.dumps(results.get("numeric_features", [])),
            train_test_split=payload.test_size,
            validation_strategy="5-Fold Cross Validation",
            mae=m["mae"],
            rmse=m["rmse"],
            r2=m["r2"],
            cv_mean_r2=m["cv_mean_r2"],
            cv_std_r2=m["cv_std_r2"],
            training_duration_ms=m["training_duration_ms"],
            feature_importances=json.dumps(m.get("feature_importances", {})),
            model_path=m["model_path"],
            is_best=is_b,
            created_at=datetime.utcnow()
        )
        db.add(ml_record)
        saved_records.append(ml_record)

    db.commit()
    return {
        "summary": results,
        "saved_models": [r.to_dict() for r in saved_records]
    }

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        p = Path(model.model_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    db.delete(model)
    db.commit()
    return {"message": "Model deleted successfully"}

# ==================== Yield Prediction ====================
@app.get("/api/predictions/requirements/{model_id}")
def get_model_requirements(model_id: str, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    features = json.loads(model.feature_columns) if model.feature_columns else []
    numerics = json.loads(model.numerical_features) if model.numerical_features else []
    cats = json.loads(model.categorical_features) if model.categorical_features else []

    # Get sample options for categorical features from dataset if available
    dataset = db.query(DatasetModel).filter(DatasetModel.id == model.dataset_id).first()
    cat_options = {}
    numeric_ranges = {}

    if dataset and dataset.meta_json:
        meta = json.loads(dataset.meta_json)
        for c in meta.get("columns", []):
            if c["name"] in cats and "top_values" in c:
                cat_options[c["name"]] = list(c["top_values"].keys())
            if c["name"] in numerics and "min" in c:
                numeric_ranges[c["name"]] = {
                    "min": c["min"],
                    "max": c["max"],
                    "default": c.get("median", c.get("mean", 50))
                }

    return {
        "model_id": model.id,
        "model_name": model.name,
        "algorithm": model.algorithm,
        "target_column": model.target_column,
        "target_unit": model.target_unit,
        "feature_columns": features,
        "numerical_features": numerics,
        "categorical_features": cats,
        "categorical_options": cat_options,
        "numeric_ranges": numeric_ranges
    }

class PredictRequest(BaseModel):
    model_id: str
    inputs: Dict[str, Any]
    notes: Optional[str] = None

@app.post("/api/predictions/predict")
def predict_yield_endpoint(payload: PredictRequest, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        result = predict_single(
            model_path=model.model_path,
            inputs=payload.inputs
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save to history
    log = PredictionLog(
        id=result["prediction_id"],
        model_id=model.id,
        model_name=model.name,
        algorithm=model.algorithm,
        inputs_json=json.dumps(payload.inputs),
        predicted_yield=result["predicted_yield"],
        unit=result["target_unit"],
        confidence_lower=result["confidence_lower"],
        confidence_upper=result["confidence_upper"],
        uncertainty_pct=result["uncertainty_pct"],
        notes=payload.notes,
        created_at=datetime.utcnow()
    )
    db.add(log)
    db.commit()

    return result

@app.get("/api/predictions/history")
def get_prediction_history(db: Session = Depends(get_db)):
    history = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(100).all()
    return [h.to_dict() for h in history]

@app.delete("/api/predictions/history/{prediction_id}")
def delete_prediction_history(prediction_id: str, db: Session = Depends(get_db)):
    pred = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(pred)
    db.commit()
    return {"message": "Prediction deleted"}

# ==================== Batch Predictions ====================
@app.post("/api/batch/preview")
async def preview_batch_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".csv", ".xlsx", ".xls"]:
        raise HTTPException(status_code=400, detail="Batch file must be CSV or Excel (.xlsx)")

    temp_id = str(uuid.uuid4())[:8]
    temp_path = UPLOAD_DIR / f"temp_batch_{temp_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        df = load_dataframe(str(temp_path))
        preview = get_preview(df, limit=20)
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Failed to read batch file: {e}")

    return {
        "temp_file_path": str(temp_path),
        "filename": file.filename,
        "total_rows": len(df),
        "columns": list(df.columns),
        "preview": preview
    }

class SuggestMappingRequest(BaseModel):
    file_columns: List[str]
    model_id: str

@app.post("/api/batch/suggest-mapping")
def suggest_mapping_endpoint(payload: SuggestMappingRequest, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    features = json.loads(model.feature_columns) if model.feature_columns else []
    mapping = suggest_column_mapping(payload.file_columns, features)
    return {
        "mapping": mapping,
        "model_features": features,
        "file_columns": payload.file_columns
    }

class RunBatchRequest(BaseModel):
    file_path: str
    model_id: str
    column_mapping: Dict[str, str]
    actual_target_column: Optional[str] = None

@app.post("/api/batch/predict")
def run_batch_endpoint(payload: RunBatchRequest, db: Session = Depends(get_db)):
    model = db.query(MLModel).filter(MLModel.id == payload.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        result = run_batch_prediction(
            file_path=payload.file_path,
            model_path=model.model_path,
            column_mapping=payload.column_mapping,
            actual_target_column=payload.actual_target_column
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Log batch
    batch_log = BatchPredictionLog(
        id=result["batch_id"],
        model_id=model.id,
        model_name=model.name,
        filename=Path(payload.file_path).name,
        result_path=result["export_csv"],
        total_records=result["total_records"],
        avg_yield=result["stats"]["mean"],
        min_yield=result["stats"]["min"],
        max_yield=result["stats"]["max"],
        unit=result["target_unit"],
        created_at=datetime.utcnow()
    )
    db.add(batch_log)
    db.commit()

    return result

@app.get("/api/batch/history")
def get_batch_history(db: Session = Depends(get_db)):
    batches = db.query(BatchPredictionLog).order_by(BatchPredictionLog.created_at.desc()).limit(50).all()
    return [b.to_dict() for b in batches]

@app.get("/api/batch/download/{filename}")
def download_batch_result(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(file_path, filename=filename)

# ==================== Agri Mall Analytics ====================
@app.get("/api/agrimall/summary")
def get_agrimall_summary():
    return parse_agrimall_data()

# ==================== Demand Forecasting ====================
@app.get("/api/forecast")
def get_demand_forecast(
    category: str = Query("ALL"),
    product: Optional[str] = Query(None),
    horizon_months: int = Query(6, ge=1, le=12),
    model_type: str = Query("holt_winters")
):
    return generate_demand_forecast(
        category=category,
        product=product,
        horizon_months=horizon_months,
        model_type=model_type
    )

# ==================== Reports ====================
class ReportRequest(BaseModel):
    title: Optional[str] = "AgriPredict AI — Agricultural Intelligence Executive Report"
    dataset_id: Optional[str] = None
    author: Optional[str] = "TY B.Sc. Data Science Research Team"

@app.post("/api/reports/generate")
def generate_report_endpoint(payload: ReportRequest, db: Session = Depends(get_db)):
    # Gather datasets, models, agrimall data
    ds_meta = None
    if payload.dataset_id:
        ds = db.query(DatasetModel).filter(DatasetModel.id == payload.dataset_id).first()
    else:
        ds = db.query(DatasetModel).filter(DatasetModel.is_active == True).first()

    if ds and ds.meta_json:
        ds_meta = json.loads(ds.meta_json)

    models = [m.to_dict() for m in db.query(MLModel).order_by(MLModel.r2.desc()).all()]
    latest_pred = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).first()
    agrimall = parse_agrimall_data()

    pdf_filename = generate_pdf_report(
        project_title=payload.title,
        dataset_summary=ds_meta,
        ml_models=models,
        latest_prediction=latest_pred.to_dict() if latest_pred else None,
        agrimall_summary=agrimall,
        author=payload.author
    )

    return {
        "filename": pdf_filename,
        "download_url": f"/api/reports/download/{pdf_filename}",
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get("/api/reports/download/{filename}")
def download_report(filename: str):
    file_path = EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(file_path, filename=filename, media_type="application/pdf")

# ==================== Frontend Production SPA Serving ====================
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("docs") or full_path.startswith("openapi.json") or full_path.startswith("redoc"):
            raise HTTPException(status_code=404, detail="Not Found")
        target_file = FRONTEND_DIST / full_path
        if target_file.exists() and target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(FRONTEND_DIST / "index.html")

