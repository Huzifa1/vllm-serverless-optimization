#!/usr/bin/env bash
set -e

# Clean up old handles
rm -f mem_handle.bin event_handle.bin

# Start A, then B
./proc_a &
PID_A=$!

# Give A a tiny head start (not strictly necessary since B waits for mem_handle.bin)
# sleep 0.2

./proc_b &
PID_B=$!

wait "$PID_A"
wait "$PID_B"
