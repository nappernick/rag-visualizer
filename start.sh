#!/bin/bash

echo "🚀 Starting RAG Visualizer System..."

# Kill any existing processes on our ports
echo "Cleaning up existing processes..."
pkill -f "uvicorn main:app.*8745" 2>/dev/null
pkill -f "vite.*5892" 2>/dev/null
pkill -f "vite.*5893" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend on port 8745..."
cd backend/src
nohup uvicorn main:app --host 0.0.0.0 --port 8745 --reload > ../../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
curl -s http://localhost:8745/health > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Backend is running on http://localhost:8745"
else
    echo "❌ Backend failed to start. Check backend.log for details"
    exit 1
fi

# Start frontend
echo "Starting frontend..."
cd ../../frontend
nohup npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
sleep 3

# Save PIDs for stop script
echo $BACKEND_PID > ../backend.pid
echo $FRONTEND_PID > ../frontend.pid

echo ""
echo "✅ System started successfully!"
echo "   Backend:  http://localhost:8745"
echo "   Frontend: http://localhost:5892 or http://localhost:5893"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop: ./stop.sh"