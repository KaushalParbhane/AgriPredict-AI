@echo off
title AgriPredict AI Desktop Launcher
set PYTHON_EXE="C:\Users\Kaushal Parbhane\AppData\Local\Programs\Orange\python.exe"

if exist %PYTHON_EXE% (
    %PYTHON_EXE% desktop_app.py
) else (
    python desktop_app.py
)
pause
