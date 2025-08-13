#!/bin/bash

echo "🚀 Starting RAG Visualizer System..."

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Kill any existing processes on our ports
echo "Cleaning up existing processes..."
pkill -f "uvicorn.*8745" 2>/dev/null
pkill -f "vite.*5892" 2>/dev/null
sleep 2

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Check for bun or npm
if command -v bun &> /dev/null; then
    PKG_MANAGER="bun"
elif command -v npm &> /dev/null; then
    PKG_MANAGER="npm"
else
    echo "❌ Neither bun nor npm is installed"
    exit 1
fi
echo "Using package manager: $PKG_MANAGER"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "backend/venv_new" ]; then
    echo "Activating virtual environment (venv_new)..."
    source backend/venv_new/bin/activate
elif [ -d "backend/venv" ]; then
    echo "Activating virtual environment (venv)..."
    source backend/venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python..."
fi

# Start backend
echo "Starting backend on port 8745..."
cd backend
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8745 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:8745/health > /dev/null 2>&1; then
        echo "✅ Backend is running on http://localhost:8745"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Backend failed to start. Check backend.log for details"
        tail -20 ../backend.log
        exit 1
    fi
done

# Start frontend
echo "Starting frontend on port 5892..."
cd ../frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    $PKG_MANAGER install
fi

$PKG_MANAGER run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
echo "Waiting for frontend to initialize..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:5892 > /dev/null 2>&1; then
        echo "✅ Frontend is running on http://localhost:5892"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "⚠️  Frontend may still be starting. Check frontend.log for status"
    fi
done

# Save PIDs for stop script
echo $BACKEND_PID > backend.pid
echo $FRONTEND_PID > frontend.pid

echo ""
echo "✅ System started successfully!"
echo "   Backend:  http://localhost:8745"
echo "   Frontend: http://localhost:5892"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop: ./stop.sh"