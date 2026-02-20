#!/usr/bin/env bash

echo "Finding GPU compute processes..."

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u)

if [[ -z "$PIDS" ]]; then
    echo "No GPU compute processes found."
    exit 0
fi

echo "Killing the following PIDs:"
echo "$PIDS"

for pid in $PIDS; do
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Killing PID $pid"
        sudo kill -9 "$pid"
    fi
done

echo "Done."
