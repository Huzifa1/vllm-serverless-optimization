#!/usr/bin/env bash

DEVICE_ID="$1"

if [[ -z "$DEVICE_ID" ]]; then
    echo "Usage: $0 <device_id>"
    exit 1
fi

if ! [[ "$DEVICE_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: device_id must be a number."
    exit 1
fi

# Get all MIG UUIDs in order
mapfile -t MIG_UUIDS < <(nvidia-smi -L | grep MIG | awk -F'UUID: ' '{print $2}' | tr -d ')')

TOTAL=${#MIG_UUIDS[@]}

if (( DEVICE_ID < 0 || DEVICE_ID >= TOTAL )); then
    echo "Error: device_id must be between 0 and $((TOTAL-1))"
    exit 1
fi

echo "${MIG_UUIDS[$DEVICE_ID]}"
