REM # scripts/dev_run.bat
@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   Starting Agentic Framework (Development)
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ERROR: Virtual environment not found!
    echo Please run scripts\setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env file exists
@REM if not exist ".env" (
@REM     echo WARNING: .env file not found!
@REM     echo Please copy .env.example to .env and configure Azure OpenAI credentials
@REM     echo.
@REM     echo Would you like to continue with default settings? (y/n)
@REM     set /p continue="Enter choice: "
@REM     if /i "!continue!" neq "y" (
@REM         echo Setup cancelled. Please configure .env file first.
@REM         pause
@REM         exit /b 1
@REM     )
@REM )

echo [1/5] Starting MCP Agent Servers...
echo.

REM Start Web Search Agent
echo Starting Web Search Agent (port 9101)...
start "Web Search Agent" cmd /k "cd /d "%cd%" && call venv\Scripts\activate.bat && python agentic\agents\local\web_search_server.py"
timeout /t 2 /nobreak >nul

REM Start Tabulator Agent  
echo Starting Tabulator Agent (port 9102)...
start "Tabulator Agent" cmd /k "cd /d "%cd%" && call venv\Scripts\activate.bat && python agentic\agents\local\tabulator_server.py"
timeout /t 2 /nobreak >nul

REM Start NLP Summarizer Agent
echo Starting NLP Summarizer Agent (port 9103)...
start "NLP Summarizer Agent" cmd /k "cd /d "%cd%" && call venv\Scripts\activate.bat && python agentic\agents\local\nlp_summarizer_server.py"
timeout /t 2 /nobreak >nul

REM Start Calculator Agent
echo Starting Calculator Agent (port 9104)...
start "Calculator Agent" cmd /k "cd /d "%cd%" && call venv\Scripts\activate.bat && python agentic\agents\local\calculator_server.py"
timeout /t 2 /nobreak >nul

echo.
echo [2/5] Waiting for agents to initialize...
timeout /t 8 /nobreak >nul

echo [3/5] Testing system...
python scripts\test_agents.py all

echo.
echo [4/4] Starting main FastAPI server...
echo.
echo ============================================
echo   🚀 Agentic Framework is starting!
echo ============================================
echo.
echo 📊 Web Interface: http://localhost:8080
echo 🔧 API Documentation: http://localhost:8080/docs
echo 🛡️ Admin Panel: Click "Admin" button in web interface
echo.
echo 📝 Available Agents:
echo   • Web Search (port 9101) - Real web search capabilities
echo   • Tabulator (port 9102) - Data processing and tables  
echo   • NLP Summarizer (port 9103) - AI text summarization
echo   • Calculator (port 9104) - Math and statistics
echo.
echo 💡 Try asking: "Do market research on electric vehicles"
echo.
echo Press Ctrl+C to stop all services
echo ============================================
echo.

REM Start main server
python -m agentic.app.main

echo.
echo ============================================
echo   Agentic Framework stopped
echo ============================================
pause