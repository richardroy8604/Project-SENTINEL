@echo off
title ULTRON AI Sentinel
cd /d "%~dp0\ULTRON"

echo ============================================================
echo   Starting ULTRON AI Sentinel (Edge Node)
echo ============================================================
echo.

if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe main.py
) else (
    python main.py
)

echo.
echo ULTRON has shut down.
pause
