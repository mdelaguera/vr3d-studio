@echo off
title Any2VR - Turn any photo or video into 3D & VR
cd /d "%~dp0"
echo ========================================================
echo   Any2VR - Turn any photo or video into 3D and VR
echo   NVIDIA CUDA accelerated
echo ========================================================
echo.
echo Starting...
python main.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the application.
    pause
)
