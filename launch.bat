@echo off
setlocal enabledelayedexpansion
title NumPy-GPT Studio Launcher
color 0A

echo ============================================================
echo               NumPy-GPT Studio Launcher
echo        Pure Python ^& NumPy Neural Language Model
echo ============================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    where py >nul 2>nul
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Python is not found in your PATH!
        echo Please install Python 3.9+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    ) else (
        set "PYTHON_CMD=py"
    )
) else (
    set "PYTHON_CMD=python"
)

echo [1/5] Checking Python environment...
%PYTHON_CMD% -c "import sys; print(f'      Found Python {sys.version.split()[0]}')"

:: Check numpy installation
%PYTHON_CMD% -c "import numpy" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] NumPy not detected. Installing dependencies from requirements.txt...
    %PYTHON_CMD% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to install NumPy. Please run: pip install numpy
        pause
        exit /b 1
    )
) else (
    echo       NumPy is installed and ready.
)

:: 2. Check Node.js and npm installation
echo.
echo [2/5] Checking Node.js environment...
where npm >nul 2>nul
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Node.js / npm is not found in your PATH!
    echo Please install Node.js 18+ from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('node -v 2^>nul') do echo       Found Node.js %%v
for /f "tokens=*" %%v in ('npm -v 2^>nul') do echo       Found npm %%v

:: 3. Check node_modules dependencies
echo.
echo [3/5] Verifying Web UI dependencies...
if not exist "node_modules\" (
    echo       Installing npm dependencies (first run only)...
    call npm install
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] npm install failed!
        pause
        exit /b 1
    )
) else (
    echo       Web UI packages already installed.
)

:: 4. Start Python Daemon
echo.
echo [4/5] Starting NumPy-GPT Daemon on port 5005...
start "NumPy-GPT Backend Daemon" /min cmd /c "%PYTHON_CMD% daemon.py"

:: Wait 2 seconds for daemon to bind port
timeout /t 2 /nobreak >nul

:: 5. Open Web Browser and Launch Vite Dev Server
echo.
echo [5/5] Launching NumPy-GPT Web Studio...
echo.
echo ============================================================
echo   Studio URL: http://localhost:3000
echo   Daemon API: http://localhost:5005
echo.
echo   * Interactive Chat ^& Autonomous Auto-Trainer are active!
echo   * Keep this window open while using the Studio.
echo   * Press Ctrl+C in this window to stop the server.
echo ============================================================
echo.

:: Open browser
start http://localhost:3000

:: Start Vite Frontend
call npm run dev

:: Clean up daemon upon exit
taskkill /FI "WINDOWTITLE eq NumPy-GPT Backend Daemon*" /F >nul 2>nul
