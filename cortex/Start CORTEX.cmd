@echo off
rem Starts the CORTEX server (if it is not already running) and opens it in the default browser.
rem The server runs in a minimized window titled "CORTEX server"; close that window to stop it.
setlocal
cd /d "%~dp0"
set "PY=%~dp0backend\.venv\Scripts\python.exe"
set "URL=http://127.0.0.1:8420"

if not exist "%PY%" (
  echo The Python environment is missing. See "Running it" in %~dp0README.md for the one-time setup.
  pause
  exit /b 1
)
if not exist "%~dp0frontend\dist\index.html" (
  echo The UI has not been built yet. Run:  cd "%~dp0frontend" ^&^& npm install ^&^& npm run build
  pause
  exit /b 1
)

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%URL%/api/health' | Out-Null; exit 0 } catch { exit 1 }"
if not errorlevel 1 goto open

echo Starting CORTEX...
start "CORTEX server" /min "%PY%" "%~dp0backend\run.py" serve
powershell -NoProfile -Command "for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%URL%/api/health' | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1"
if errorlevel 1 (
  echo CORTEX did not start within 30 seconds. Check the "CORTEX server" window for the error.
  pause
  exit /b 1
)

:open
start "" "%URL%"
