@echo off
title ULTRON Monitor Station -- Web Companion Hub
cd /d "%~dp0"

echo ============================================================
echo   Starting ULTRON Monitor Station (Dashboard & MQTT Hub)
echo ============================================================

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" server.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Monitor station stopped with error code %ERRORLEVEL%.
    pause
)
