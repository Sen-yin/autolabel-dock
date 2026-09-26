@echo off
REM Run AutoLabel Dock with the project venv (no need to manually activate).
setlocal
cd /d "%~dp0"
".\venv\Scripts\python.exe" main.py %*
endlocal