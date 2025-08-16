# scripts/setup.sh
#!/bin/bash

set -e

echo "==========================================="
echo "  Agentic Framework Setup (Linux/Mac)"
echo "==========================================="

echo ""
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3.10+ is required but not found."
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $REQUIRED_VERSION or higher is required. Found $PYTHON_VERSION"
    exit 1
fi

echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "[3/6] Activating virtual environment..."
source venv/bin/activate

echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip

echo "[5/6] Installing dependencies..."
pip install -r requirements.txt

echo "[6/6] Creating directory structure..."
mkdir -p agentic/agents
mkdir -p agentic/prompts
mkdir -p agentic/workflows  
mkdir -p agentic/app/static
mkdir -p logs
mkdir -p tests

echo ""
echo "Creating environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Please edit .env file with your Azure OpenAI credentials"
fi

echo ""
echo "============================================"
echo "  Setup completed successfully!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your credentials"
echo "2. Run: ./scripts/dev_run.sh"
echo "3. Open browser to http://localhost:8080"
echo ""



