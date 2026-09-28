@echo off
title ULTRON Terminal Monitor -- Live MQTT Feed
cd /d "%~dp0"

echo ============================================================
echo   Starting ULTRON Terminal Monitor (Live MQTT Stream)
echo ============================================================

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" subscriber_cli.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    pause
)
