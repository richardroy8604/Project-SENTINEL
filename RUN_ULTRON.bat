@echo off
echo ============================================================
echo   U L T R O N  —  AI Security System
echo ============================================================
echo.
echo Starting ULTRON...
echo.
cd /d "%~dp0"
venv\Scripts\python.exe main.py
echo.
echo ULTRON has shut down.
pause
