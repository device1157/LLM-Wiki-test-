@echo off
setlocal

cd /d "%~dp0"

echo Starting Chinese History Wiki...
echo.

where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found.
        echo Install Python 3.10 or newer, then run this file again.
        echo Download: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python virtual environment...
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3 -m venv .venv
    )
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        echo Trying python directly...
        python -m venv .venv
        if errorlevel 1 (
            echo.
            echo Could not create .venv. Install Python 3.10+ and try again.
            pause
            exit /b 1
        )
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo Could not activate .venv.
    pause
    exit /b 1
)

python -c "import streamlit, openai, yaml" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Package installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo.
echo Opening Chinese History Wiki at http://localhost:8501
echo Close this window to stop the app.
echo.
echo Optional add-ons:
echo   - Anthropic Claude provider:  pip install anthropic
echo   - Local BERT Chinese NER:     pip install transformers torch
echo     (Entity extraction otherwise uses your configured LLM or a built-in
echo      dynasty gazetteer, so the app works without these.)
echo.

start "" "http://localhost:8501"
python -m streamlit run app.py

echo.
echo Chinese History Wiki stopped.
pause
