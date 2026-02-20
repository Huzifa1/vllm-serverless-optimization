export CUDA_VISIBLE_DEVICES="0"

NUM_PROMPTS=${1:-100}
USE_CUSTOM_SCHEDULER=${2:-true}
USE_MIG=${3:-true}
USE_CUSTOM_TIMESTAMPS=${4:-true}
MODEL_NAME=${5:-"llama3-3b"}
PORT=${6:-8000}

MAIN_DIR="$(cd "$(dirname "$(readlink -f "$0")")"/../.. && pwd)" # vllm-serverless-optimization
MODELS_DIR="$MAIN_DIR/models"
SECOND_PARENT_NAME="$(basename "$(dirname "$(dirname "$(readlink -f "$0")")")")" # new_idea
MODEL_TO_PORT_FILE="$MAIN_DIR/$SECOND_PARENT_NAME/model_to_port_map.json"
OUTPUT_FILE="$MAIN_DIR/$SECOND_PARENT_NAME/logs/bench_results_${MODEL_NAME}_${NUM_PROMPTS}.json"
DATASET_FILE="$MAIN_DIR/$SECOND_PARENT_NAME/traces/modified_conversation_trace_${NUM_PROMPTS}.jsonl"

if [ "$USE_CUSTOM_SCHEDULER" = true ] || [ "$USE_MIG" = true ]; then
  cmd=(
    vllm bench serve
    --backend vllm
    --endpoint /v1/completions
    --num-prompts "$NUM_PROMPTS"
    --save_result
    --result_filename "$OUTPUT_FILE"
    --percentile_metrics 'ttft,tpot,itl'
    --metric_percentiles '95,99'
    --dataset-name custom
    --dataset-path "$DATASET_FILE"
    --custom-skip-chat-template
    --use_timestamp
    --model-to-port-map-file "$MODEL_TO_PORT_FILE"
  )

  # Conditionally add flags
  if [[ "$USE_CUSTOM_SCHEDULER" == true ]]; then
    cmd+=(--custom-scheduler)
  fi

  if [[ "$USE_MIG" == true ]]; then
    cmd+=(--use_mig)
  fi

  # Run command
  "${cmd[@]}"
else
  if [ "$USE_CUSTOM_TIMESTAMPS" = true ]; then
    vllm bench serve \
      --backend vllm \
      --model "$MODELS_DIR/$MODEL_NAME" \
      --endpoint /v1/completions \
      --num-prompts $NUM_PROMPTS \
      --save_result \
      --result_filename $OUTPUT_FILE \
      --percentile_metrics 'ttft,tpot,itl' \
      --metric_percentiles '95,99' \
      --dataset-name custom \
      --dataset-path $DATASET_FILE \
      --custom-skip-chat-template \
      --use_timestamp
  else
    # Without scheduler
    vllm bench serve \
      --backend vllm \
      --model "$MODELS_DIR/$MODEL_NAME" \
      --endpoint /v1/completions \
      --num-prompts $NUM_PROMPTS \
      --save_result \
      --result_filename $OUTPUT_FILE \
      --percentile_metrics 'ttft,tpot,itl' \
      --metric_percentiles '95,99' \
      --dataset-name sharegpt \
      --dataset-path ../traces/ShareGPT_V3_unfiltered_cleaned_split.json \
      --port $PORT
  fi
fi