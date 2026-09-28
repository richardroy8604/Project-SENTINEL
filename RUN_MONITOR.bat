@echo off
title ULTRON Monitor Station
cd /d "%~dp0\MONITOR_APP"

echo ============================================================
echo   Starting ULTRON Monitor Station (Dashboard & MQTT Hub)
echo ============================================================
echo.

if exist "..\venv\Scripts\python.exe" (
    ..\venv\Scripts\python.exe server.py
) else (
    python server.py
)

echo.
echo Monitor station has shut down.
pause
