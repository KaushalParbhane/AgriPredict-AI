import json
from fastapi.testclient import TestClient
from backend.main import app

def run_tests():
    client = TestClient(app)

    print("1. Testing GET /api/health...")
    res = client.get("/api/health")
    assert res.status_code == 200, res.text
    print("   -> OK:", res.json()["status"])

    print("2. Testing GET /api/overview...")
    res = client.get("/api/overview")
    assert res.status_code == 200, res.text
    overview = res.json()
    print("   -> OK: Datasets =", overview["kpis"]["total_datasets"], "Records =", overview["kpis"]["total_records"])

    print("3. Testing GET /api/models...")
    res = client.get("/api/models")
    assert res.status_code == 200, res.text
    models = res.json()
    print(f"   -> OK: {len(models)} trained models available")
    best_model = [m for m in models if m.get("is_best")][0]
    print(f"   -> Best model: {best_model['name']} (R2: {best_model['r2']})")

    print("4. Testing POST /api/predictions/predict...")
    sample_inputs = {
        "District": "Pune",
        "Crop": "Wheat",
        "Season": "Rabi",
        "Rainfall_mm": 380.0,
        "Temperature_C": 22.5,
        "Soil_Type": "Black Soil",
        "Soil_pH": 7.1,
        "Nitrogen_N": 105.0,
        "Phosphorus_P": 52.0,
        "Potassium_K": 41.0,
        "Irrigation_Type": "Drip Irrigation",
        "Fertilizer_Used_kg": 230.0
    }
    res = client.post("/api/predictions/predict", json={
        "model_id": best_model["id"],
        "inputs": sample_inputs,
        "notes": "Test field prediction in Pune region"
    })
    assert res.status_code == 200, res.text
    pred = res.json()
    print(f"   -> OK: Predicted Yield = {pred['predicted_yield']} {pred['target_unit']}, Range: [{pred['confidence_lower']} - {pred['confidence_upper']}]")

    print("5. Testing GET /api/agrimall/summary...")
    res = client.get("/api/agrimall/summary")
    assert res.status_code == 200, res.text
    mall = res.json()
    print(f"   -> OK: Mall: {mall['mall_name']}, Revenue: Rs.{mall['total_revenue']:,.2f}, Categories: {len(mall['categories'])}")

    print("6. Testing GET /api/forecast...")
    res = client.get("/api/forecast?category=SEEDS&horizon_months=6")
    assert res.status_code == 200, res.text
    fc = res.json()
    print(f"   -> OK: Forecast model = {fc['model_used']}, Points = {len(fc['forecast_data'])}, MAPE = {fc['metrics']['mape_pct']}%")

    print("7. Testing POST /api/reports/generate...")
    res = client.post("/api/reports/generate", json={"title": "AgriPredict Capstone Verification Report"})
    assert res.status_code == 200, res.text
    rep = res.json()
    print(f"   -> OK: PDF generated = {rep['filename']}")

    print("8. Testing GET / (SPA index)...")
    res = client.get("/")
    assert res.status_code == 200, res.text
    assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()
    print("   -> OK: Frontend SPA served successfully at root /")

    print("\nALL AUTOMATED INTEGRATION TESTS PASSED WITH 100% SUCCESS!")

if __name__ == "__main__":
    run_tests()
