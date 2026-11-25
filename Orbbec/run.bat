@echo off
rem Laptop: double-click. First run installs everything (about 3 GB download), later runs start at once.
rem run.bat left = the closet is on the left of the picture (default: right)
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
set SIDE=%1
if "%SIDE%"=="" set SIDE=right
.venv\Scripts\python live.py --side %SIDE%
pause
