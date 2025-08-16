# scripts/build.sh
#!/bin/bash

echo "Building Agentic Framework"
echo "=========================="

source venv/bin/activate

echo "Running tests..."
python -m pytest tests/ -v

echo "Building package..."
python -m build

echo "Build completed successfully!"


