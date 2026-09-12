#!/usr/bin/env bash
set -e

echo "============================================================"
echo "              NumPy-GPT Studio Launcher"
echo "       Pure Python & NumPy Neural Language Model"
echo "============================================================"
echo ""

# 1. Python check
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found in PATH! Please install Python 3.9+"
    exit 1
fi

echo "[1/4] Checking Python environment..."
python3 -c "import sys; print(f'      Found Python {sys.version.split()[0]}')"

if ! python3 -c "import numpy" &> /dev/null; then
    echo "[INFO] Installing requirements..."
    python3 -m pip install -r requirements.txt
fi

# 2. Node check
echo "[2/4] Checking Node.js environment..."
if ! command -v npm &> /dev/null; then
    echo "[ERROR] npm not found! Please install Node.js from https://nodejs.org/"
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "      Installing web packages..."
    npm install
fi

# 3. Start Daemon
echo "[3/4] Starting Python daemon on port 5005..."
python3 daemon.py > /tmp/numpy_gpt_daemon.log 2>&1 &
DAEMON_PID=$!

cleanup() {
    echo ""
    echo "[STOP] Terminating backend daemon (PID $DAEMON_PID)..."
    kill $DAEMON_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2

# 4. Launch Vite
echo "[4/4] Launching Studio UI on http://localhost:3000..."
echo "============================================================"
echo "  Studio URL: http://localhost:3000"
echo "  Daemon API: http://localhost:5005"
echo "  Press Ctrl+C to stop both frontend and backend."
echo "============================================================"

# Try opening default browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000 &> /dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:3000 &> /dev/null &
fi

npm run dev
