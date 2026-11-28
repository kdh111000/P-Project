@echo off
rem What GitHub does not get, for moving to another PC by USB: this PC's own 26 (models\held.pt, fine-tuned on its closet),
rem the other 26s, registered garments, camweb recordings and checked scenes, ATR cut-outs (more_data.py synth), the page key.
rem -> fitmate_local.tar here. On the other PC, in ClosetCam after git clone:  tar -xf fitmate_local.tar
setlocal enabledelayedexpansion
cd /d "%~dp0"
set L=
for %%p in (models registered db_cache doorcam_web collect yolo\raw\data camweb_key.txt) do if exist "%%p" set L=!L! "%%p"
if "!L!"=="" (echo Nothing to pack here. & pause & exit /b 1)
tar -cf fitmate_local.tar !L! && echo fitmate_local.tar ready:!L!
pause
