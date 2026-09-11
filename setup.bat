@echo off
REM One-time setup for Windows. Double-click this file, or run: setup.bat
REM Then start the app with run.bat
setlocal
cd /d "%~dp0"

echo.
echo === 1/3  Checking Python ===
REM The py launcher ships with the python.org installer and is the reliable
REM way to find a real Python on Windows -- a bare "python" can be the Store
REM stub that opens the Microsoft Store instead of running anything.
set "PY="
py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo   Python 3.9 or newer was not found.
  echo   Install it from https://www.python.org/downloads/
  echo   IMPORTANT: tick "Add python.exe to PATH" in the installer.
  echo.
  pause
  exit /b 1
)
%PY% --version

echo.
echo === 2/3  Installing Python packages into .venv ===
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :failed
echo   ok  packages installed

echo.
echo === 3/3  Checking Tesseract (needed only to read Form 6 scans) ===
where tesseract >nul 2>&1
if %errorlevel%==0 (
  echo   ok  found on PATH
  goto :done
)
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" (
  echo   ok  found in %ProgramFiles%\Tesseract-OCR
  echo       The app detects it there automatically.
  goto :done
)
echo   !   Tesseract is not installed.
where winget >nul 2>&1
if %errorlevel%==0 (
  echo.
  set /p REPLY="      Install it now with winget? [y/N] "
  if /i "%REPLY%"=="y" (
    winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
    goto :done
  )
)
echo       Download it from https://github.com/UB-Mannheim/tesseract/wiki
echo       Everything except the Form 6 upload works without it.

:done
echo.
echo Done. Start the app by double-clicking run.bat
echo.
pause
exit /b 0

:failed
echo.
echo   Installing the packages failed. Check the messages above.
echo.
pause
exit /b 1
