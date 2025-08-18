REM # scripts/dev_run.bat
@echo off
echo ===========================================
echo  Starting Agentic Framework (Development)
echo ===========================================

call venv\Scripts\activate.bat

echo Starting MCP agent servers...
start "Web Search Agent" cmd /k "fastmcp run agentic\agents\local\web_search_server.py --transport http --port 9101"
timeout /t 2 /nobreak >nul

start "Tabulator Agent" cmd /k "fastmcp run agentic\agents\local\tabulator_server.py --transport http --port 9102"
timeout /t 2 /nobreak >nul

start "NLP Summarizer Agent" cmd /k "fastmcp run agentic\agents\local\nlp_summarizer_server.py --transport http --port 9103"
timeout /t 2 /nobreak >nul

start "Calculator Agent" cmd /k "fastmcp run agentic\agents\local\calculator_server.py --transport http --port 9104"
timeout /t 2 /nobreak >nul

echo Waiting for agents to start...
timeout /t 5 /nobreak >nul

echo Starting main server...
echo Open browser to: http://localhost:8080
python -m agentic.app.main


