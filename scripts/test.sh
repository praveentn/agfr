# scripts/test.sh
#!/bin/bash

echo "Running Agentic Framework Tests"
echo "================================"

source venv/bin/activate

echo "Running unit tests..."
python -m pytest tests/ -v --tb=short

echo ""
echo "Running integration tests..."
python -m pytest tests/integration/ -v --tb=short

echo ""
echo "Running type checking..."
python -m mypy agentic/

echo ""
echo "Running code formatting check..."
python -m black --check agentic/
python -m isort --check-only agentic/

