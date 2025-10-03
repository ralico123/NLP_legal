#!/bin/bash

echo "�� Starting Legal Document Summarizer..."
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

echo "🔧 Creating necessary directories..."
mkdir -p uploads

echo "🌐 Starting web server..."
echo "The application will be available at: http://localhost:8080"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
