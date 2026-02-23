#!/bin/bash
# Kill any process on port 5001, then start the dashboard (ensures latest code is loaded)
cd "$(dirname "$0")/.."
echo "Checking port 5001..."
if command -v lsof >/dev/null 2>&1; then
  PID=$(lsof -ti :5001 2>/dev/null)
  if [ -n "$PID" ]; then
    echo "Killing process $PID on port 5001..."
    kill -9 $PID 2>/dev/null
    sleep 2
  fi
fi
echo "Starting dashboard on http://127.0.0.1:5001 ..."
exec python3 -m src.serve_dashboard
