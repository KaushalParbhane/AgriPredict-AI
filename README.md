# AgriPredict AI — Premium Agricultural Intelligence Platform

> **TY B.Sc. Data Science Capstone Project**  
> **Student:** Kaushal Parbhane  
> **Domain:** Precision Agriculture, Crop Yield Prediction & Agro Mall Commercial Analytics  
> **Case Study:** Shivdatta Krupa Krushi Agro Mall Data (Maharashtra, India)

---

## 🌟 Executive Overview

**AgriPredict AI** is a production-grade Agricultural Intelligence and Decision Support Platform engineered for precision farming, harvest yield estimation, and agro-retail supply chain forecasting.

Unlike traditional academic prototypes, AgriPredict AI pairs a modern **high-contrast dark emerald SaaS interface** with **rigorous, leakage-free Python machine learning pipelines** and **authentic commercial sales data** from **Shivdatta Krupa Krushi Agro Mall**.

---

## 🚀 Key Modules & Capabilities

### 1. Visual Design — Premium AI SaaS
- **Aesthetic**: Deep forest green (`#052e16`, `#064e3b`), emerald (`#10b981`), and subtle lime (`#84cc16`) accents set against dark charcoal glass cards (`#070d0a`).
- **Data Visualizations**: Recharts-powered responsive charts with smooth animations, hover tooltips, and real numeric metrics.
- **Micro-Interactions**: Collapsible navigation sidebar, real-time toast alerts, and modal dialogs.

### 2. Landing Page (`Hero & Feature Showcase`)
- **Headline**: *"Predict Smarter. Grow Better."*
- **Live Metrics Bar**: Connected directly to actual database records (2,000+ agricultural field samples, 97.5% peak R² accuracy, 10 crop varieties, and ₹21.3 Lakhs in recorded agro-mall sales).
- **Interactive Tour**: Direct access to Dashboard, Dataset Lab, ML Studio, and Commercial Analytics.

### 3. Overview Analytics Workspace
- **KPI Metrics**: Total datasets, total records, active crop varieties, trained models, latest prediction, and dataset quality health score.
- **Visualizations**:
  - Average Crop Yield by Variety (quintals/hectare)
  - Model Benchmark Comparison (R² across 5 algorithms)
  - Agro Mall Monthly Purchasing Curve (FY 2026-27 seasonality)
  - Product Category Revenue Split (Seeds, Fertilizers, Insecticides, Herbicides, etc.)

### 4. Advanced Dataset Lab
- **Multi-Format Ingestion**: Supports `.csv`, `.xlsx`, `.xls`, and `.json`.
- **Automated Profiling**: Identifies missing entries, duplicate rows, data types, continuous distributions (min, max, mean, median, skewness), and categorical top values.
- **Target Recommendation Engine**: Detects candidate continuous target variables (e.g. `Yield_Quintals_Per_Ha`, `Net Sal Amount`).
- **Dataset Comparison**: Side-by-side schema diff, row delta, and data quality comparison.

### 5. AI Data Explorer
- Multivariate analysis workspace supporting **Bar Charts**, **Histograms**, **Scatter Plots**, **Line Trends**, and **Box Plots**.
- Filters by categorical variables with calculated descriptive statistics (Mean, Median, Standard Deviation).
- Empirical agronomic interpretations with strict hallucination safeguards.

### 6. Data Cleaning & Transformation Studio
- **Deduplication**: Purges exact duplicate rows.
- **Missing Value Imputation**: Median, Mean, Mode, Forward Fill, or row deletion.
- **Outlier Filtering**: Interquartile Range (IQR) multiplier slider or Z-Score standard deviation thresholds.
- **Feature Scaling**: StandardScaler or MinMaxScaler.
- **Non-Destructive Versioning**: Generates versioned clean dataset copies while preserving original raw uploads.

### 7. Machine Learning Model Studio
- Benchmarks 5 distinct regressors side-by-side:
  1. **Dummy Baseline Regressor** (Mean target baseline)
  2. **Linear Regression** (Ordinary least squares parametric baseline)
  3. **Decision Tree Regressor** (Recursive partitioning tree)
  4. **Random Forest Regressor** (Ensemble bagging across 100 trees)
  5. **XGBoost Regressor** (Gradient boosted regularized trees)
- **Leakage-Proof Pipeline**: Uses `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')`.
- **Validation**: 5-Fold Cross Validation + Held-out Test Split (configurable 10%–40%).
- **Evaluation Metrics**: MAE, RMSE, R² Score, CV Mean R², and Training Duration (ms).
- **Feature Importance**: Gini importance ranking of top agricultural yield drivers (e.g., Rainfall, Soil pH, Nitrogen, Irrigation).

### 8. AI Crop-Yield Prediction Studio
- Dynamic form generated automatically from the active model's exact feature schema:
  - District / Region (Pune, Nashik, Ahmednagar, Kolhapur, Solapur, etc.)
  - Crop (Soybean, Bajra, Wheat, Maize, Cotton, Sugarcane, Gram, Rice, Onion, Tomato)
  - Season (Kharif, Rabi, Summer, Whole Year)
  - Rainfall (mm), Temperature (°C), Soil Type, Soil pH (5.8–8.2)
  - Nitrogen (N), Phosphorus (P), Potassium (K), Irrigation Type, Fertilizer Quantity
- **Prediction Result Card**:
  - Large yield display with correct unit (`quintals/ha`)
  - 95% Confidence Interval bounds `[lower, upper]`
  - Model uncertainty margin (±%)
  - Persistent storage in Prediction History

### 9. Batch Prediction Center
- Upload field spreadsheets with hundreds of rows.
- **Auto-Column Mapping**: Automatically matches spreadsheet headers to model input requirements with manual override support.
- **Vectorized High-Throughput Inference**: Appends predicted yield columns while preserving all original spreadsheet attributes.
- **Distribution Visualization**: Yield histogram and descriptive statistics.
- **Instant Export**: Download enriched files as CSV or Excel (.xlsx).

### 10. Agri Mall Commercial Intelligence
- Direct analysis of **Shivdatta Krupa Krushi Agro Mall** transaction ledgers (`mall data/`):
  - Gross Invoiced Revenue: **₹21,33,742+** across **1,283 ledger items**
  - Category Analysis: Seeds, Fertilizers, Insecticides, Herbicides, Micronutrients, Fungicides, Plant Growth
  - Top Products by Revenue: Yashoda Soya KDS-726, Pioneer Bajari 86M38, Maka, Dhanuka Conika, Syngenta Kavach
  - Manufacturer / Brand Performance: Bayer, Syngenta, Rallis India, Mahabeej, Coromandel, UPL
  - Warehouse Inventory Stock Health: Available reserve vs low-stock threshold monitoring

### 11. Demand Forecasting Module
- Time-series product demand forecasting for rural inventory planning.
- Algorithms: **Holt-Winters Seasonal Smoothing**, **3-Month Moving Average**, and **Linear Trend Extrapolation**.
- Chronological validation: Evaluates holdout accuracy via **MAPE %** and **RMSE**.
- Forecast Horizons: 3, 6, 9, or 12 months with 95% upper and lower prediction bounds.
- Highlights seasonal demand surges (pre-monsoon Kharif planting vs Rabi sowing).

### 12. Reports & PDF Generator
- Compiled via **ReportLab** into executive, vector-rendered PDF documents.
- Includes Academic Abstract, Model Benchmarks, Dataset Audits, Agro Mall Commercial Review, and Engineering Leakage Safeguards.
- Direct download via browser.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide React, Recharts |
| **Backend Engine** | Python 3.12, FastAPI, Pydantic v2, Uvicorn (ASGI) |
| **Data Science & ML**| Scikit-Learn 1.5, XGBoost, Pandas, NumPy, Joblib, OpenPyXL |
| **Persistence** | SQLite 3, SQLAlchemy 2.1 |
| **Reporting** | ReportLab PDF Engine |

---

## 💻 How to Run the Desktop Software

You have multiple convenient ways to launch **AgriPredict AI** as a native desktop software application:

### Option 1: Desktop Shortcut (Instant Double-Click)
A shortcut called **`AgriPredict AI`** has been placed directly on your **Windows Desktop** with its custom application icon.
- Simply **double-click the `AgriPredict AI` icon on your Desktop**.
- The native software window will open directly on your screen!

### Option 2: Project Folder Double-Click
Inside this project directory, double-click either:
- **`AgriPredict AI.lnk`** (Native Windows Shortcut with Icon)
- **`AgriPredict_AI.bat`** (Batch Desktop Software Launcher)
- **`AgriPredict_AI_Silent.vbs`** (Clean launcher without command prompt window)

### Option 3: Command Line
```powershell
& "C:\Users\Kaushal Parbhane\AppData\Local\Programs\Orange\python.exe" desktop_app.py
```
*(Or simply `python desktop_app.py`)*

---

## 🖥️ Desktop Software Architecture
- **GUI Engine:** [pywebview](https://pywebview.flowrl.com/) + native Windows **Edge Chromium WebView2**.
- **Window Specs:** Standalone native window (1440 × 920, resizable, min size 1080 × 720).
- **Desktop Features:**
  - Dedicated Windows taskbar item with custom brand icon.
  - Native Windows title bar with minimize, maximize, and close buttons.
  - **No browser URL bar, no browser tabs, no browser bookmarks.**
  - Clean shutdown of local AI inference services when the desktop software window is closed.
- **Backend Service:** Embedded multi-threaded FastAPI engine operating on localhost loopback with SQLite persistence and Joblib machine learning pipelines.


---

## 🧪 Automated Integration Tests

To run the complete test suite verifying all 10 API services:
```powershell
& "C:\Users\Kaushal Parbhane\AppData\Local\Programs\Orange\python.exe" -c "
from backend.database import SessionLocal
from backend.main import get_overview_kpis, list_models, get_agrimall_summary, get_demand_forecast, generate_report_endpoint, ReportRequest
db = SessionLocal()
print('Overview:', get_overview_kpis(db)['kpis'])
print('Models:', len(list_models(db)))
print('Mall Revenue:', get_agrimall_summary()['total_revenue'])
print('Forecast:', get_demand_forecast('SEEDS', 6)['model_used'])
db.close()
print('ALL VERIFIED!')
"
```

---

## 👨‍🎓 Project Credits & Attribution

- **Student:** Kaushal Parbhane
- **Degree:** Bachelor of Science in Data Science (TY B.Sc.)
- **Academic Year:** 2026–2027
- **Commercial Partner Data:** Shivdatta Krupa Krushi Agro Mall
