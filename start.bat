@echo off
echo =================================================================
echo   Starting AgriPredict AI Platform
echo =================================================================
set PYTHON_EXE="C:\Users\Kaushal Parbhane\AppData\Local\Programs\Orange\python.exe"
if exist %PYTHON_EXE% (
    %PYTHON_EXE% run_app.py
) else (
    python run_app.py
)
pause
