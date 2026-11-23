import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CLEANED_DIR = DATA_DIR / "cleaned"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = BASE_DIR / "saved_models"
MALL_DATA_DIR = PROJECT_ROOT / "mall data"

# Ensure directories exist
for directory in [DATA_DIR, UPLOAD_DIR, CLEANED_DIR, EXPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{BASE_DIR / 'agripredict.db'}"
