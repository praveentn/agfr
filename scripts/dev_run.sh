# scripts/dev_run.sh
#!/bin/bash

echo "==========================================="
echo "  Starting Agentic Framework (Development)"
echo "==========================================="

source venv/bin/activate

echo "Starting MCP agent servers..."

# Start agent servers in background
python agentic/agents/local/web_search_server.py &
WEB_SEARCH_PID=$!

python agentic/agents/local/tabulator_server.py &
TABULATOR_PID=$!

python agentic/agents/local/nlp_summarizer_server.py &
NLP_PID=$!

python agentic/agents/local/calculator_server.py &
CALC_PID=$!

# Wait for agents to start
echo "Waiting for agents to start..."
sleep 5

# Function to cleanup background processes
cleanup() {
    echo "Stopping agent servers..."
    kill $WEB_SEARCH_PID $TABULATOR_PID $NLP_PID $CALC_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

echo "Starting main server..."
echo "Open browser to: http://localhost:8080"
python -m agentic.app.main



