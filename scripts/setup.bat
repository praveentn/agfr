REM # scripts/setup.bat
@echo off
echo ===========================================
echo  Agentic Framework Setup (Windows)
echo ===========================================

echo.
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10+ is required but not found.
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
)

echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat

echo [4/6] Upgrading pip...
python -m pip install --upgrade pip

echo [5/6] Installing dependencies...
pip install -r requirements.txt

echo [6/6] Creating directory structure...
if not exist "agentic\agents" mkdir agentic\agents
if not exist "agentic\prompts" mkdir agentic\prompts
if not exist "agentic\workflows" mkdir agentic\workflows
if not exist "agentic\app\static" mkdir agentic\app\static
if not exist "logs" mkdir logs
if not exist "tests" mkdir tests

echo.
echo Creating environment file...
if not exist ".env" (
    copy .env.example .env
    echo Please edit .env file with your Azure OpenAI credentials
)

echo.
echo ============================================
echo  Setup completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Edit .env file with your credentials
echo 2. Run: scripts\dev_run.bat
echo 3. Open browser to http://localhost:8080
echo.
pause

