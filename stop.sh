#!/bin/bash

echo "🛑 Stopping RAG Visualizer System..."

# Stop using PIDs if available
if [ -f backend.pid ]; then
    BACKEND_PID=$(cat backend.pid)
    echo "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    rm backend.pid
else
    echo "No backend.pid found, killing by port..."
    pkill -f "uvicorn main:app.*8745" 2>/dev/null
fi

if [ -f frontend.pid ]; then
    FRONTEND_PID=$(cat frontend.pid)
    echo "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null
    rm frontend.pid
else
    echo "No frontend.pid found, killing by port..."
    pkill -f "vite.*5892" 2>/dev/null
fi

# Also kill by port pattern as fallback
echo "Cleaning up any remaining processes..."
pkill -f "uvicorn main:app.*8745" 2>/dev/null
pkill -f "vite.*5892" 2>/dev/null
lsof -ti:8745 | xargs kill -9 2>/dev/null
lsof -ti:5892 | xargs kill -9 2>/dev/null
lsof -ti:5893 | xargs kill -9 2>/dev/null

echo "✅ System stopped"
echo ""
echo "📝 Check logs if needed:"
echo "   Backend:  cat backend.log"
echo "   Frontend: cat frontend.log"