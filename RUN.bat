@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Fruit Ripeness Assessment

set "ENVROOT=%LOCALAPPDATA%\FruitRipeness"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"
set "YOLO_CONFIG_DIR=%ENVROOT%"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo.
    echo  The app is not set up on this computer yet.
    echo  Running SETUP.bat first - this needs an internet connection.
    echo.
    pause
    call "%~dp0SETUP.bat"
    if not exist "%PY%" (
        echo.
        echo  Setup did not complete, so the app cannot start.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%~dp0streamlit_app.py" (
    echo.
    echo  [X] streamlit_app.py is missing. Keep RUN.bat in the project
    echo      folder, next to streamlit_app.py.
    echo.
    pause
    exit /b 1
)

if /i "%~1"=="desktop" goto DESKTOP

set "PORT=8501"
:FINDPORT
netstat -an -p tcp | findstr /r /c:"^ *TCP *[^ ]*:%PORT% " >nul
if errorlevel 1 goto PORTREADY
set /a PORT+=1
if %PORT% LEQ 8599 goto FINDPORT
set "PORT=8501"
:PORTREADY

echo.
echo ===================================================================
echo   Fruit Ripeness Assessment
echo ===================================================================
echo   Starting... your browser will open at http://localhost:%PORT%
echo   The first analysis takes a few seconds while the model loads.
echo   Leave this window open while you use the app.
echo   Press Ctrl+C or close this window to stop.
echo ===================================================================
echo.

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:%PORT%/_stcore/health | Out-Null; Start-Process http://localhost:%PORT%; break } catch { Start-Sleep -Seconds 1 } }"

"%PY%" -m streamlit run streamlit_app.py --server.port %PORT%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ===================================================================
    echo   The app stopped with an error ^(code %RC%^).
    echo.
    echo   Run SETUP.bat again - it re-checks every package and prints
    echo   what is wrong.
    echo ===================================================================
    echo.
    pause
)
exit /b %RC%

:DESKTOP
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" "%~dp0fruit_pipeline_gui.py"
exit /b 0
