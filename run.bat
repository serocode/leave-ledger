@echo off
REM Start the app. Double-click this file. Run setup.bat first if you haven't.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   Not set up yet. Double-click setup.bat first.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" app.py
pause
