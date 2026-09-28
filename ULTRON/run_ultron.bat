@echo off
title ULTRON AI Sentinel -- Edge Node
cd /d "%~dp0"

echo ============================================================
echo   Starting ULTRON AI Sentinel (Edge Node)
echo ============================================================

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] ULTRON stopped with error code %ERRORLEVEL%.
    pause
)
