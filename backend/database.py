import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    is_cleaned = Column(Boolean, default=False)
    parent_id = Column(String(36), nullable=True)
    meta_json = Column(Text, default="{}")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "file_type": self.file_type,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
            "is_cleaned": self.is_cleaned,
            "parent_id": self.parent_id,
            "meta": json.loads(self.meta_json) if self.meta_json else {}
        }


class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    algorithm = Column(String(64), nullable=False)
    dataset_id = Column(String(36), nullable=False)
    target_column = Column(String(128), nullable=False)
    target_unit = Column(String(64), default="quintals/ha")
    feature_columns = Column(Text, default="[]")
    categorical_features = Column(Text, default="[]")
    numerical_features = Column(Text, default="[]")
    train_test_split = Column(Float, default=0.2)
    validation_strategy = Column(String(64), default="5-Fold Cross Validation")
    mae = Column(Float, default=0.0)
    rmse = Column(Float, default=0.0)
    r2 = Column(Float, default=0.0)
    cv_mean_r2 = Column(Float, default=0.0)
    cv_std_r2 = Column(Float, default=0.0)
    training_duration_ms = Column(Float, default=0.0)
    feature_importances = Column(Text, default="{}")
    model_path = Column(String(512), nullable=False)
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "algorithm": self.algorithm,
            "dataset_id": self.dataset_id,
            "target_column": self.target_column,
            "target_unit": self.target_unit,
            "feature_columns": json.loads(self.feature_columns) if self.feature_columns else [],
            "categorical_features": json.loads(self.categorical_features) if self.categorical_features else [],
            "numerical_features": json.loads(self.numerical_features) if self.numerical_features else [],
            "train_test_split": self.train_test_split,
            "validation_strategy": self.validation_strategy,
            "mae": round(self.mae, 4) if self.mae is not None else None,
            "rmse": round(self.rmse, 4) if self.rmse is not None else None,
            "r2": round(self.r2, 4) if self.r2 is not None else None,
            "cv_mean_r2": round(self.cv_mean_r2, 4) if self.cv_mean_r2 is not None else None,
            "cv_std_r2": round(self.cv_std_r2, 4) if self.cv_std_r2 is not None else None,
            "training_duration_ms": round(self.training_duration_ms, 2) if self.training_duration_ms else 0.0,
            "feature_importances": json.loads(self.feature_importances) if self.feature_importances else {},
            "is_best": self.is_best,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True)
    model_id = Column(String(36), nullable=False)
    model_name = Column(String(255), nullable=False)
    algorithm = Column(String(64), nullable=False)
    inputs_json = Column(Text, nullable=False)
    predicted_yield = Column(Float, nullable=False)
    unit = Column(String(64), default="quintals/ha")
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    uncertainty_pct = Column(Float, default=5.0)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "inputs": json.loads(self.inputs_json) if self.inputs_json else {},
            "predicted_yield": round(self.predicted_yield, 2),
            "unit": self.unit,
            "confidence_lower": round(self.confidence_lower, 2) if self.confidence_lower is not None else None,
            "confidence_upper": round(self.confidence_upper, 2) if self.confidence_upper is not None else None,
            "uncertainty_pct": round(self.uncertainty_pct, 1),
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class BatchPredictionLog(Base):
    __tablename__ = "batch_predictions"

    id = Column(String(36), primary_key=True)
    model_id = Column(String(36), nullable=False)
    model_name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    result_path = Column(String(512), nullable=False)
    total_records = Column(Integer, default=0)
    avg_yield = Column(Float, default=0.0)
    min_yield = Column(Float, default=0.0)
    max_yield = Column(Float, default=0.0)
    unit = Column(String(64), default="quintals/ha")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "filename": self.filename,
            "total_records": self.total_records,
            "avg_yield": round(self.avg_yield, 2),
            "min_yield": round(self.min_yield, 2),
            "max_yield": round(self.max_yield, 2),
            "unit": self.unit,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
