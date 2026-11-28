@echo off
rem Double-click: the camweb page (phone or laptop camera in a browser) + the local Qwen in its own window.
rem   run.bat live [left^|right]   the older way: this PC's camera in a window (live.py)
rem First run installs everything (about 3 GB), qwen.bat about 5.8 GB more; later runs start at once.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\ok (
  where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
  if errorlevel 1 (
    echo Python 3.10 - 3.13 is needed: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install -U pip
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Check the internet connection and run this again.
    pause
    exit /b 1
  )
  git lfs pull 2>nul
  echo ok> .venv\ok
)
if /i "%1"=="live" (
  if "%2"=="" (.venv\Scripts\python live.py --side right) else (.venv\Scripts\python live.py --side %2)
  pause
  exit /b
)
if /i not "%FITMATE_VL%"=="0" start "Qwen3-VL" cmd /c qwen.bat
.venv\Scripts\python camweb.py
pause
