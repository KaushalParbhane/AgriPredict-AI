"""
AgriPredict AI — Startup Script
Runs the unified full-stack server serving both the FastAPI REST backend and the React SPA frontend.
"""
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    python_exe = sys.executable

    print("=" * 65)
    print("  AgriPredict AI — Premium Agricultural Intelligence Platform")
    print("  TY B.Sc. Data Science Capstone Project")
    print("=" * 65)
    print("\n[1/2] Initializing SQLite database & verification...")

    # Run seed script
    subprocess.run([python_exe, "-m", "backend.seed_data"], check=True)

    print("\n[2/2] Launching AgriPredict AI Application on http://localhost:8000...")
    print("      • Open your browser to: http://localhost:8000")
    print("      • REST API Documentation: http://localhost:8000/docs")
    print("      • Press Ctrl+C to stop the server\n")

    # Run Uvicorn serving backend.main:app
    subprocess.run([
        python_exe, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    main()
