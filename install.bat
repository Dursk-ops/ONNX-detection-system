@echo off
title DurskAI Installer
color 0F
cd /d "%~dp0pythonruntimes"
chcp 65001 >nul

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
echo          DurskAI - Installer
echo.
echo  ============================================
echo.
echo  This will install these packages globally:
echo    - onnxruntime
echo    - opencv-python
echo    - numpy
echo    - mss
echo    - pillow
echo    - customtkinter
echo    - vgamepad
echo.
echo  Press any key to begin...
pause >nul

REM ─────────────────────────────────────────────────────────────
REM  Step 1: Check Python
REM ─────────────────────────────────────────────────────────────
cls
echo.
echo  [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python is not installed or not on PATH.
    echo.
    echo  Download Python 3.11 or 3.12 from:
    echo    https://www.python.org/downloads/windows/
    echo  During install, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo        Python %PYVER% detected.
timeout /t 1 /nobreak >nul

REM ─────────────────────────────────────────────────────────────
REM  Step 2: Upgrade pip
REM ─────────────────────────────────────────────────────────────
echo.
echo  [2/3] Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo        pip up to date.
timeout /t 1 /nobreak >nul

REM ─────────────────────────────────────────────────────────────
REM  Step 3: Install dependencies globally
REM ─────────────────────────────────────────────────────────────
echo.
echo  [3/3] Installing packages (this may take a minute)...
echo.

python -m pip install ^
    onnxruntime ^
    opencv-python ^
    numpy ^
    mss ^
    pillow ^
    customtkinter ^
    vgamepad

if errorlevel 1 (
    echo.
    echo  ERROR: One or more packages failed to install.
    echo.
    echo  Common fixes:
    echo    - Use Python 3.11 or 3.12 (3.13+ may lack wheels)
    echo    - Run: python -m pip install --upgrade pip
    echo    - Run this file as Administrator
    echo.
    pause
    exit /b 1
)

REM ─────────────────────────────────────────────────────────────
REM  Verify
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
echo  Verifying install...
echo.

python -c "import cv2, numpy, onnxruntime, mss, customtkinter, PIL; print('  Core packages: OK')"
python -c "import vgamepad; print('  vgamepad    : OK')" 2>nul
if errorlevel 1 (
    echo   vgamepad    : FAILED - gamepad mode will not work
) else (
    echo   vgamepad    : OK  - ViGEmBus driver installs on first use
)

echo.
echo  ============================================
echo   INSTALL COMPLETE
echo  ============================================
echo.
echo   Next steps:
echo     - Double-click run.bat to launch DurskAI
echo     - Put your .onnx model in the models\ folder
echo     - First gamepad use triggers a ViGEmBus
echo       driver install popup - approve it.
echo.
echo  Press any key to close...
pause >nul