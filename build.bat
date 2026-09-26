@echo off
REM Build a Windows distributable of AutoLabel Dock with PyInstaller.
REM
REM Usage (from repo root):
REM     build.bat
REM
REM Outputs:
REM     dist\AutoLabelDock\AutoLabelDock.exe          (launcher)
REM     dist\AutoLabelDock\_internal\...             (bundled deps + .ico)
REM
REM Optional environment overrides (set BEFORE calling build.bat):
REM     PYI_FLAGS=--clean --noconfirm                extra pyinstaller flags
REM     VENV_DIR=venv                                 venv to use for pyinstaller
REM
REM Prerequisites:
REM     1. Project venv exists (created via README instructions) and contains
REM        PyInstaller + the project's runtime deps.
REM     2. resources\icons\app.ico exists. If missing, this script regenerates
REM        it from resources\icons\app_icon.png via tools\build_icon.py.

setlocal EnableDelayedExpansion

REM ── Paths ──────────────────────────────────────────────────────────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
if not defined VENV_DIR set "VENV_DIR=%ROOT%\venv"
set "PY=%VENV_DIR%\Scripts\python.exe"

REM ── Pre-flight ────────────────────────────────────────────────────────
if not exist "%PY%" (
    echo [build] ERROR: Python not found at %PY%
    echo         Create the venv first:  python -m venv venv
    echo         Then:                  venv\Scripts\pip install -r requirements.txt pyinstaller
    exit /b 1
)

REM Refresh the .ico from the source PNG if it's stale or missing.
set "ICON_PNG=%ROOT%\resources\icons\app_icon.png"
set "ICON_ICO=%ROOT%\resources\icons\app.ico"
if not exist "%ICON_ICO%" (
    if not exist "%ICON_PNG%" (
        echo [build] ERROR: %ICON_PNG% not found.
        echo         Drop the source PNG into resources\icons\app_icon.png and re-run.
        exit /b 1
    )
    echo [build] Generating app.ico from app_icon.png ...
    "%PY%" "%ROOT%\tools\build_icon.py" || exit /b 1
)

REM ── Sanity: pyinstaller installed? ───────────────────────────────────
"%PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [build] PyInstaller is not installed in %VENV_DIR%.
    echo         Installing now ...
    "%PY%" -m pip install pyinstaller || exit /b 1
)

REM ── Clean previous build ─────────────────────────────────────────────
if exist "%ROOT%\build"   rmdir /s /q "%ROOT%\build"
if exist "%ROOT%\dist"    rmdir /s /q "%ROOT%\dist"

REM ── Run pyinstaller ──────────────────────────────────────────────────
echo [build] Running pyinstaller ...
"%PY%" -m PyInstaller --noconfirm autolabel.spec %PYI_FLAGS%
if errorlevel 1 (
    echo [build] pyinstaller failed.
    exit /b 1
)

REM ── Patch torchvision natives ───────────────────────────────────────
REM PyInstaller's static analyzer doesn't follow ``import torchvision``
REM into its C-extension .pyd files or bundled runtime DLLs. Without
REM them ultralytics raises ``operator torchvision::nms does not
REM exist`` the first time NMS runs. The patch script copies them from
REM the venv into ``dist\AutoLabelDock\_internal\torchvision\``.
echo [build] Patching torchvision natives ...
"%PY%" "%ROOT%\packaging\patch_torchvision.py" --exe-dir "%ROOT%\dist\AutoLabelDock"
if errorlevel 1 (
    echo [build] WARNING: torchvision patch incomplete — training may fail with NMS errors.
)

echo.
echo [build] Done.
echo         Launcher: %ROOT%\dist\AutoLabelDock\AutoLabelDock.exe
echo         Bundle:   %ROOT%\dist\AutoLabelDock\_internal\
echo.
echo         Tip: zip the entire AutoLabelDock\ folder when distributing.

endlocal
