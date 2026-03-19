@echo off
title SGD Portable
setlocal EnableDelayedExpansion
cd /d "%~dp0"
echo SGD Portable Start %date% %time% > install.log 2>&1
echo.
echo  SGD Portable is starting...
echo  For problems: check install.log in this folder.
echo.

set "PYTHON_DIR=%~dp0python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip"
set "PYTHON_ZIP=%~dp0py_embed.zip"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "GETPIP=%PYTHON_DIR%\get-pip.py"
set "PACKAGES_OK=%PYTHON_DIR%\packages_ok.txt"
set "MODEL_DIR=%~dp0maest_model"

if exist "%PYTHON_EXE%" goto pip_check

echo [1/4] Downloading Python 3.12...
curl -L --progress-bar -o "%PYTHON_ZIP%" "%PYTHON_URL%"
if not exist "%PYTHON_ZIP%" (
    echo ERROR: Python download failed.
    pause & exit /b 1
)
echo Extracting Python...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force" >> install.log 2>&1
del "%PYTHON_ZIP%" 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "if (Test-Path '%PYTHON_DIR%\python312._pth') { (Get-Content '%PYTHON_DIR%\python312._pth') -replace '#import site','import site' | Set-Content '%PYTHON_DIR%\python312._pth' }" >> install.log 2>&1
if not exist "%PYTHON_DIR%\Lib\site-packages" mkdir "%PYTHON_DIR%\Lib\site-packages"
echo Python OK

:pip_check
if exist "%PYTHON_DIR%\Scripts\pip.exe" goto packages_check
echo [2/4] Installing pip...
curl -L --progress-bar -o "%GETPIP%" "%GETPIP_URL%"
if not exist "%GETPIP%" (
    echo ERROR: pip download failed.
    pause & exit /b 1
)
"%PYTHON_EXE%" "%GETPIP%" --no-warn-script-location >> install.log 2>&1
if errorlevel 1 (
    echo ERROR: pip installation failed. See install.log
    pause & exit /b 1
)
del "%GETPIP%" 2>nul
echo pip OK

:packages_check
if exist "%PACKAGES_OK%" goto model_check

echo [3/4] Installing packages (one-time setup)...
echo.
echo  Packages to download and install:
echo.
echo    torch (CPU)                ~450 MB   AI engine
echo    transformers + deps        ~250 MB   MAEST model loader
echo    librosa + audio + deps     ~120 MB   audio processing
echo    gradio + plotly + deps     ~400 MB   web interface and charts
echo    pip-intern                 ~430 MB   cache, metadata, bytecode
echo.
echo  Total: ~2 GB  (packages + MAEST model ~350 MB, downloaded once)
echo.
echo  Please be patient, this can take a while.
echo.

echo [3a] Installing torch (CPU, ~450 MB)...
"%PYTHON_EXE%" -m pip install --no-warn-script-location --target="%PYTHON_DIR%\Lib\site-packages" torch --index-url https://download.pytorch.org/whl/cpu >> install.log 2>&1
if errorlevel 1 ( echo ERROR: torch failed. See install.log & pause & exit /b 1 )
echo     torch OK

echo [3b] Installing remaining packages (~800 GB)...
"%PYTHON_EXE%" -m pip install --no-warn-script-location --target="%PYTHON_DIR%\Lib\site-packages" transformers librosa soundfile numpy gradio plotly >> install.log 2>&1
if errorlevel 1 ( echo ERROR: Package installation failed. See install.log & pause & exit /b 1 )
echo     packages OK

echo packages_ok > "%PACKAGES_OK%"
echo.
echo  All packages installed successfully.

:model_check
if exist "%MODEL_DIR%\hub\models--mtg-upf--discogs-maest-10s-fs-129e" goto launch

echo [4/4] Downloading MAEST model (~350 MB, one-time)...
echo  This is the AI model for genre detection.
echo.
"%PYTHON_EXE%" -c "import os,sys,warnings; warnings.filterwarnings('ignore'); p=sys.argv[1]; os.environ['HF_HOME']=p; os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'; os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'; from huggingface_hub import snapshot_download; snapshot_download('mtg-upf/discogs-maest-10s-fs-129e')" "%MODEL_DIR%" >> install.log 2>&1
if errorlevel 1 (
    echo WARNING: Model download failed. It will be downloaded on first use.
) else (
    echo     MAEST model OK
)

:launch
echo.
echo  Starting SGD Portable...
echo  Leave this window open. Close it to stop the app.
echo.
"%PYTHON_EXE%" "%~dp0genre_webui.py"
echo.
echo App finished.
pause
