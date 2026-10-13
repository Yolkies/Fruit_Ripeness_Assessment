@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Fruit Ripeness Assessment - Setup

set "ENVROOT=%LOCALAPPDATA%\FruitRipeness"
set "RT=%ENVROOT%\rt"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "YOLO_CONFIG_DIR=%ENVROOT%"
set "PYTHONIOENCODING=utf-8"
set "PROBE=%TEMP%\fruitripeness_probe.txt"
set "MODEL=runs\segment\Fruit_Project\segmentation_run\weights\best.pt"

set "TORCH_VERSION=2.11.0"
set "TORCHVISION_VERSION=0.26.0"
set "CPU_INDEX=https://download.pytorch.org/whl/cpu"
set "GPU_INDEX=https://download.pytorch.org/whl/cu126"
set "PYURL=https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

set "TORCHMODE=ask"
if /i "%~1"=="gpu" set "TORCHMODE=gpu"
if /i "%~1"=="cpu" set "TORCHMODE=cpu"

echo.
echo ===================================================================
echo   Fruit Ripeness Assessment - Setup
echo ===================================================================
echo   Project     : %CD%
echo   Environment : %VENV%
echo.

if not exist "%MODEL%" (
    echo [X] %MODEL% is missing.
    echo     Run SETUP.bat from inside the project folder, and make sure
    echo     the whole project was downloaded or extracted.
    goto FAILED
)

echo [1/5] Looking for Python 3.10-3.12...

set "BASEPY="
call :FINDPY py -3.12
if not defined BASEPY call :FINDPY py -3.11
if not defined BASEPY call :FINDPY py -3.10
if not defined BASEPY call :FINDPY python
if not defined BASEPY call :FINDPY "%RT%\python.exe"

if defined BASEPY (
    echo       Using: !BASEPY!
    goto HAVEPYTHON
)

echo       No suitable Python found - downloading a private copy ^(~21 MB^).
echo       Note: it is not digitally signed, so Windows Smart App Control may
echo       block it. Installing Python 3.12 from python.org and running
echo       SETUP.bat again avoids that.

where curl.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS
where tar.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS

if not exist "%ENVROOT%" mkdir "%ENVROOT%"
curl.exe -L --fail --retry 3 --retry-delay 2 -o "%ENVROOT%\python.tar.gz" "%PYURL%"
if errorlevel 1 (
    echo [X] Download failed. Check your internet connection and try again.
    goto FAILED
)

echo       Extracting...
tar.exe -xzf "%ENVROOT%\python.tar.gz" -C "%ENVROOT%"
if errorlevel 1 (
    echo [X] Could not extract the downloaded Python.
    goto FAILED
)
if exist "%ENVROOT%\python\python.exe" (
    if exist "%RT%" rmdir /s /q "%RT%"
    move /y "%ENVROOT%\python" "%RT%" >nul
)
del /q "%ENVROOT%\python.tar.gz" >nul 2>&1

call :FINDPY "%RT%\python.exe"
if not defined BASEPY (
    echo [X] The downloaded Python did not unpack as expected.
    goto FAILED
)
echo       Using: !BASEPY!

:HAVEPYTHON
echo.

if exist "%PY%" (
    set "VBASE="
    set "NBASE="
    "%PY%" -c "import sys;sys.stdout.write(sys.base_prefix)" > "%PROBE%" 2>nul
    set /p VBASE=<"%PROBE%"
    "%BASEPY%" -c "import sys;sys.stdout.write(sys.prefix)" > "%PROBE%" 2>nul
    set /p NBASE=<"%PROBE%"
    if /i "!VBASE!"=="!NBASE!" (
        echo [2/5] Virtual environment already exists - reusing it.
        goto HAVEVENV
    )
    echo [2/5] The existing environment was built on a different Python - rebuilding.
    rmdir /s /q "%VENV%" 2>nul
    if exist "%VENV%" (
        echo [X] The old environment could not be removed. Close the app
        echo     and any window running it, then run SETUP.bat again.
        goto FAILED
    )
)

echo [2/5] Creating the virtual environment...
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 (
    echo [X] Could not create the virtual environment at:
    echo     %VENV%
    goto FAILED
)

:HAVEVENV
if not exist "%PY%" (
    echo [X] The virtual environment has no python.exe at:
    echo     %PY%
    goto FAILED
)
"%PY%" -c "import _multiprocessing, ssl, ctypes" 2>nul
if errorlevel 1 (
    echo [X] The environment cannot load core Python modules.
    echo     This is usually Windows Smart App Control blocking an unsigned
    echo     Python. Install Python 3.12 from https://www.python.org/downloads/
    echo     then delete %ENVROOT% and run SETUP.bat again.
    goto FAILED
)
"%PY%" --version

echo.
echo [3/5] Preparing pip...
"%PY%" -m ensurepip --upgrade >nul 2>&1
"%PY%" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --upgrade pip
if errorlevel 1 (
    echo [X] Could not prepare pip inside the virtual environment.
    goto FAILED
)

echo.
echo [4/5] Installing packages
if /i not "%TORCHMODE%"=="ask" goto INSTALLTORCH

"%PY%" -c "import torch,sys;sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 (
    set "TORCHMODE=gpu"
    goto INSTALLTORCH
)

set "TORCHMODE=cpu"
nvidia-smi -L >nul 2>&1
if errorlevel 1 goto INSTALLTORCH

set "GPUNAME="
for /f "delims=" %%G in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do if not defined GPUNAME set "GPUNAME=%%G"
echo       NVIDIA graphics card found: !GPUNAME!
echo.
echo       The app itself runs fine on the CPU. The GPU build of PyTorch
echo       only makes training a new model much faster. It is a much
echo       bigger download: about 2.5 GB instead of about 200 MB.
choice /c YN /n /m "      Install the GPU build of PyTorch? [Y/N] "
if not errorlevel 2 set "TORCHMODE=gpu"

:INSTALLTORCH
if /i "%TORCHMODE%"=="gpu" (
    set "TORCHINDEX=%GPU_INDEX%"
    set "TORCHTAG=cu126"
) else (
    set "TORCHINDEX=%CPU_INDEX%"
    set "TORCHTAG=cpu"
)

echo.
echo       PyTorch %TORCH_VERSION% ^(!TORCHTAG!^)...
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location --default-timeout=120 --retries 5 "torch==%TORCH_VERSION%+!TORCHTAG!" "torchvision==%TORCHVISION_VERSION%+!TORCHTAG!" --index-url "!TORCHINDEX!"
if errorlevel 1 goto PIPFAILED

echo.
echo       Ultralytics YOLO, OpenCV, Streamlit and the rest...
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location --default-timeout=120 --retries 5 -r requirements.txt
if errorlevel 1 goto PIPFAILED

echo.
echo [5/5] Checking the installation...
"%PY%" -c "import cv2, numpy, PIL, reportlab, streamlit, torch, ultralytics; from ultralytics import YOLO, settings; settings.update({'sync': False}); YOLO(r'%MODEL%'); print('      torch', torch.__version__, '/ ultralytics', ultralytics.__version__, '/ streamlit', streamlit.__version__, '/ CUDA', torch.cuda.is_available())"
if errorlevel 1 (
    echo.
    echo [X] The packages are installed but the check above failed.
    echo     Run SETUP.bat again. If it keeps failing, delete
    echo     %ENVROOT%
    echo     and run SETUP.bat once more.
    goto FAILED
)

"%PY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo       Note: Tkinter is missing, so "RUN.bat desktop" will not work.
    echo       The browser version is not affected.
)

echo.
echo ===================================================================
echo   SETUP COMPLETE
echo.
echo   Now double-click RUN.bat to start the app.
echo ===================================================================
echo.
pause
exit /b 0


:FINDPY
if defined BASEPY goto :eof
set "CAND="
%* -c "import sys;sys.stdout.write(sys.executable if (3,10)<=sys.version_info[:2]<(3,13) and sys.maxsize>2**32 else '')" > "%PROBE%" 2>nul
if errorlevel 1 goto :eof
for %%S in ("%PROBE%") do if %%~zS EQU 0 goto :eof
set /p CAND=<"%PROBE%"
if not defined CAND goto :eof
set "BASEPY=%CAND%"
goto :eof


:NOTOOLS
echo [X] No Python was found, and curl.exe / tar.exe are not available to
echo     download one. Install Python 3.12 from
echo     https://www.python.org/downloads/ and run SETUP.bat again.
goto FAILED

:PIPFAILED
echo.
echo [X] Package installation failed, usually because the internet
echo     connection dropped. Run SETUP.bat again - it continues from
echo     where it stopped.
goto FAILED

:FAILED
echo.
echo ===================================================================
echo   SETUP FAILED - see the message above.
echo ===================================================================
echo.
pause
exit /b 1
