@echo off
title Cat vs Dog API Server
echo ===================================================
echo   Cat vs Dog MLOps Engine
echo ===================================================
echo.

:: Change to the directory where the .bat file is located
cd /d "%~dp0"

echo [*] Activating virtual environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [!] Virtual environment not found. Please run setup first.
    pause
    exit /b 1
)

echo [*] Opening API Documentation in default browser...
:: Wait 2 seconds before opening the browser to give the server time to start
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000

echo [*] Starting FastAPI Server...
echo [i] Press CTRL+C to stop the server
echo.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

echo.
pause
