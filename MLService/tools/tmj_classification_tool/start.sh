#!/bin/bash

# TMJ Classification Tool - Startup Script

echo "🦷 TMJ Classification Tool"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: Please run this script from the tmj_classification_tool directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="../../../venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install/update dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r ../../requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting server..."
echo "Open http://localhost:8000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the app
python3 app.py
