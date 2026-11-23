"""
AgriPredict AI — Native Desktop Application
Launches AgriPredict AI as a standalone Windows desktop software window
(powered by pywebview and native Windows Edge WebView2 engine).
"""
import sys
import os
import time
import socket
import threading
from pathlib import Path

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.seed_data import seed_database_if_needed

def is_port_in_use(port: int = 8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_backend_server():
    import uvicorn
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def launch_desktop():
    print("=" * 65)
    print("  AgriPredict AI — Native Desktop Application")
    print("  TY B.Sc. Data Science Software")
    print("=" * 65)

    # 1. Ensure Database & Models are seeded
    print("[1/3] Checking agricultural datasets & ML pipelines...")
    seed_database_if_needed()

    # 2. Check if backend is already running on port 8000
    if not is_port_in_use(8000):
        print("[2/3] Initializing local AI inference engine in background thread...")
        server_thread = threading.Thread(target=start_backend_server, daemon=True)
        server_thread.start()
        # Wait up to 5 seconds for backend to start listening
        for _ in range(25):
            if is_port_in_use(8000):
                break
            time.sleep(0.2)
    else:
        print("[2/3] Connected to active local AI inference engine on port 8000.")

    # 3. Launch native desktop GUI window
    print("[3/3] Opening native desktop software window...")
    import webview

    icon_path = str(PROJECT_ROOT / "agripredict.ico")
    if not os.path.exists(icon_path):
        icon_path = None

    window = webview.create_window(
        title="AgriPredict AI — Agricultural Intelligence Platform",
        url="http://127.0.0.1:8000",
        width=1440,
        height=920,
        min_size=(1080, 720),
        resizable=True,
        fullscreen=False,
        background_color="#060b08",
        confirm_close=False
    )

    # Start native Windows GUI loop (blocking until window is closed)
    webview.start(debug=False)
    print("Software window closed. Exiting cleanly.")

if __name__ == "__main__":
    launch_desktop()
