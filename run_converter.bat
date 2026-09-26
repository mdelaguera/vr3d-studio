@echo off
title VR3D Studio - 2D to 3D SBS & VR180 Converter
cd /d "%~dp0"
echo ========================================================
echo   VR3D Studio - AI 2D to 3D SBS and VR180 Converter
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
