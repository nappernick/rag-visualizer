#!/bin/bash
# Force restart script to pick up new environment variables

echo "🔄 Force restarting with fresh environment..."

# Kill all python processes related to uvicorn
pkill -f "uvicorn.*main:app" 2>/dev/null
pkill -f "python.*uvicorn" 2>/dev/null
sleep 2

# Clear Python cache
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "✅ Environment cleared, ready for restart"