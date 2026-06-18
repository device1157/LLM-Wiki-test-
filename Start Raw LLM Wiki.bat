@echo off
setlocal

cd /d "%~dp0"

echo Starting Raw LLM Wiki...
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

python -c "import streamlit, openai, anthropic" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages...
    python -m pip install streamlit openai anthropic
    if errorlevel 1 (
        echo.
        echo Package installation failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo.
echo Opening Raw LLM Wiki at http://localhost:8501
echo Close this window to stop the app.
echo.

start "" "http://localhost:8501"
python -m streamlit run app.py

echo.
echo Raw LLM Wiki stopped.
pause
