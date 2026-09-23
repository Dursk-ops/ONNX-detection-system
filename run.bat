@echo off
title DurskAI Launcher
color 0F
cd /d "%~dp0pythonruntimes"
chcp 65001 >nul

REM ─────────────────────────────────────────────────────────────
REM  DurskAI launcher
REM ─────────────────────────────────────────────────────────────

cls
echo.
echo  ============================================
echo.
echo     D D D   U   U   R R R   S S S   K   K
echo     D    D  U   U   R    R  S       K  K
echo     D    D  U   U   R R R    S S    K K
echo     D    D  U   U   R   R        S  K  K
echo     D D D    U U    R    R  S S S   K   K
echo.
echo                  A   I
echo.
echo  ============================================
echo.
echo          DurskAI - Initializing
echo.
echo  ============================================
echo.

echo  Press any key to start...
pause >nul

cls
echo.
echo  ============================================
echo.
echo     D D D   U   U   R R R   S S S   K   K
echo     D    D  U   U   R    R  S       K  K
echo     D    D  U   U   R R R    S S    K K
echo     D    D  U   U   R   R        S  K  K
echo     D D D    U U    R    R  S S S   K   K
echo.
echo                  A   I
echo.
echo  ============================================
echo.

echo  [1/6] Preparing environment...
timeout /t 1 /nobreak >nul

echo  [2/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Run install.bat first, or install Python 3.11/3.12.
    echo.
    pause
    exit /b 1
)
timeout /t 1 /nobreak >nul

echo  [3/6] Loading modules...
timeout /t 1 /nobreak >nul

echo  [4/6] Reading config...
timeout /t 1 /nobreak >nul

echo  [5/6] Starting AI engine...
timeout /t 1 /nobreak >nul

echo  [6/6] Launching GUI...
timeout /t 1 /nobreak >nul

echo.
echo  ============================================
echo   Ready.
echo  ============================================
echo.
echo  Press any key to launch DurskAI...
pause >nul

cls
echo.
echo  Launching DurskAI...
echo.
python aimaim.py

if errorlevel 1 (
    echo.
    echo  ============================================
    echo    The app exited with an error.
    echo    Screenshot the text above and send it.
    echo  ============================================
    pause
)