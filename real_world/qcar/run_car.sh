#!/bin/bash
# Run yolo_server.py and the vehicle controller together on the QCar.
#
# Usage:  ./run_car.sh [controller.py]
#   e.g.  ./run_car.sh taxi_control.py --log run1.csv
#
# Defaults to the autonomous taxi.
#
# Ctrl+C stops BOTH cleanly. The controller installs a SIGINT handler that
# zeroes throttle and steering on the way out, so it must be stopped with
# SIGINT -- never SIGKILL, or the car drives away with the script gone.

set -u

# --- settings (match config.txt / start.py) ----------------------------------
CONTROLLER="${1:-taxi_control.py}"
# Everything after the controller name is passed straight through to it, e.g.
#   ./run_car.sh taxi_control.py --log run1.csv --cruise 0.45
if [ $# -gt 0 ]; then shift; fi
CONTROLLER_ARGS=("$@")
PROBING="${PROBING:-True}"
HOST_IP="${HOST_IP:-192.168.2.24}"
WIDTH="${WIDTH:-320}"
HEIGHT="${HEIGHT:-200}"

cd "$(dirname "$(readlink -f "$0")")" || exit 1

# Both scripts do "from utils import ..." so they must run from this directory.
if [ ! -f "$CONTROLLER" ]; then
    echo "ERROR: $CONTROLLER not found in $(pwd)"
    echo "Available:"; ls -1 *.py 2>/dev/null | sed 's/^/  /'
    exit 1
fi
if [ ! -f yolo_server.py ]; then
    echo "ERROR: yolo_server.py not found in $(pwd)"; exit 1
fi

PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then echo "ERROR: no python interpreter found"; exit 1; fi

LOGDIR="${LOGDIR:-$(pwd)/logs}"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)

YOLO_PID=""
CTRL_PID=""

stop_all() {
    trap '' INT TERM                     # don't re-enter while shutting down
    echo ""
    echo "Stopping (SIGINT so the controller can zero the motor)..."

    # Controller first, so the car is commanded to stop before anything else.
    if [ -n "$CTRL_PID" ] && kill -0 "$CTRL_PID" 2>/dev/null; then
        kill -INT "$CTRL_PID" 2>/dev/null
        for _ in $(seq 1 50); do                  # up to 5 s for a clean exit
            kill -0 "$CTRL_PID" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$CTRL_PID" 2>/dev/null; then
            echo "WARNING: controller ignored SIGINT, forcing it down."
            echo "         CHECK THE CAR HAS ACTUALLY STOPPED."
            kill -KILL "$CTRL_PID" 2>/dev/null
        fi
    fi

    if [ -n "$YOLO_PID" ] && kill -0 "$YOLO_PID" 2>/dev/null; then
        kill -INT "$YOLO_PID" 2>/dev/null
        for _ in $(seq 1 30); do
            kill -0 "$YOLO_PID" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$YOLO_PID" 2>/dev/null
    fi

    echo "Both stopped. Logs in $LOGDIR"
    exit 0
}
trap stop_all INT TERM

echo "Working dir : $(pwd)"
echo "Python      : $PY"
echo "Controller  : $CONTROLLER ${CONTROLLER_ARGS[*]}"
echo "Logs        : $LOGDIR"
echo ""

# YOLO server first: the controller sleeps 10 s at startup waiting for it,
# and YOLOReceiver connects to localhost:18666.
# NB: process substitution, not "| tee". In a pipeline $! is the PID of the
# LAST element (tee), and signalling tee would leave python -- and the car --
# running.
"$PY" -u yolo_server.py -p "$PROBING" -i "$HOST_IP" -w "$WIDTH" -ht "$HEIGHT" \
    > >(tee "$LOGDIR/yolo_$STAMP.log") 2>&1 &
YOLO_PID=$!
echo "Started yolo_server.py       (pid $YOLO_PID)"

sleep 1

"$PY" -u "$CONTROLLER" "${CONTROLLER_ARGS[@]}" \
    > >(tee "$LOGDIR/control_$STAMP.log") 2>&1 &
CTRL_PID=$!
echo "Started $CONTROLLER (pid $CTRL_PID)"
echo ""
echo "Press Ctrl+C to stop both."
echo ""

# If either one dies on its own, bring the other down too -- a controller
# left running without the YOLO server will not react to signs or people.
while true; do
    if ! kill -0 "$CTRL_PID" 2>/dev/null; then
        echo "Controller exited."; stop_all
    fi
    if ! kill -0 "$YOLO_PID" 2>/dev/null; then
        echo "YOLO server exited -- stopping the controller too."; stop_all
    fi
    sleep 0.5
done
