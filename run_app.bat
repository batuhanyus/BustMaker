@echo off
setlocal enableextensions enabledelayedexpansion

:: Change directory to the root of the project where this batch file lives
cd /d "%~dp0"

echo ===================================================
echo               Starting Bust Forge App             
echo ===================================================

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in .venv
    echo.
    echo Please set up the environment first by running:
    echo   py -3.12 -m venv .venv
    echo   .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat

:: Launch the Gradio Web GUI application
echo Launching Gradio Web GUI (app.py)...
echo App will be accessible at http://127.0.0.1:7860
echo.
python app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] app.py exited with error code %errorlevel%.
)

pause
