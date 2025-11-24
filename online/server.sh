# Important. However, this exposes sensitive endpoints (e.g., /sleep, /wakeup)
export VLLM_SERVER_DEV_MODE=1
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOGGING_CONFIG_PATH="../vllm_logging_config.json"

# Read from args. By default, set to true
ENABLE_SLEEP_MODE=${1:-true}
MODEL=${2:-"../models/llama3-3b"}
LOG_PATH=${3:-"./server.log"}
PORT=${4:-8500}

if [ "$ENABLE_SLEEP_MODE" = true ] ; then
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PORT \
        --enable-sleep-mode >> $LOG_PATH 2>&1
else
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --port $PORT >> $LOG_PATH 2>&1
fi