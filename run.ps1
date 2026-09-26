# Run AutoLabel Dock with the project venv (no need to manually activate).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
& ".\venv\Scripts\python.exe" main.py @args