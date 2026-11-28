@echo off
rem Local Qwen3-VL for the unsure calls (vl.py), on 127.0.0.1:8090. First run downloads llama.cpp (Vulkan: AMD, NVIDIA, Intel)
rem and the model, about 5.8 GB, into llamacpp\. Another llama.cpp folder: set LLAMACPP=C:\path\llamacpp before running.
cd /d "%~dp0"
if "%LLAMACPP%"=="" set LLAMACPP=%~dp0llamacpp
set HF=https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main
if not exist "%LLAMACPP%\bin\llama-server.exe" (
  mkdir "%LLAMACPP%\bin" 2>nul
  curl -fL -o "%LLAMACPP%\llama.zip" https://github.com/ggml-org/llama.cpp/releases/download/b12176/llama-b12176-bin-win-vulkan-x64.zip || goto fail
  tar -xf "%LLAMACPP%\llama.zip" -C "%LLAMACPP%\bin" || goto fail
  del "%LLAMACPP%\llama.zip"
)
mkdir "%LLAMACPP%\models" 2>nul
for %%f in (Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf) do (
  if not exist "%LLAMACPP%\models\%%f" (
    curl -fL -o "%LLAMACPP%\models\%%f.part" %HF%/%%f && move /y "%LLAMACPP%\models\%%f.part" "%LLAMACPP%\models\%%f" >nul || goto fail
  )
)
rem -ngl 99: every layer on the GPU (8 GB). Less GPU memory: lower it. Answers like "GGGG..." mean the server went bad: restart it.
"%LLAMACPP%\bin\llama-server.exe" -m "%LLAMACPP%\models\Qwen3VL-8B-Instruct-Q4_K_M.gguf" --mmproj "%LLAMACPP%\models\mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf" -ngl 99 -c 8192 -np 1 --host 127.0.0.1 --port 8090
exit /b
:fail
echo Download failed. Check the internet connection and run qwen.bat again.
pause
exit /b 1
