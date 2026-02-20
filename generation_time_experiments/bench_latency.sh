
MODEL_NAME=${1:-"qwen-4b"}
OUTPUT_PATH=${2:-"output.log"}
BATCH_SIZE=${3:-1}

MAIN_DIR="$(cd "$(dirname "$(readlink -f "$0")")"/.. && pwd)" # vllm-serverless-optimization
MODELS_DIR="$MAIN_DIR/models"
PARENT_FOLDER="$(basename "$(dirname "$(readlink -f "$0")")")"

export CUDA_VISIBLE_DEVICES="0"

vllm bench latency \
  --model "$MODELS_DIR/$MODEL_NAME" \
  --dtype float16 \
  --batch-size $BATCH_SIZE \
  --input-len 1024 \
  --output-len 256 \
  --greedy_decoding \
  --n 1 | tee $OUTPUT_PATH

# Remove all lines except the results (last 7 lines)
tail -n 7 $OUTPUT_PATH > tmp.tmp
mv tmp.tmp $OUTPUT_PATH