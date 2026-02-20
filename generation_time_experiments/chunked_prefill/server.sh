MAX_TOKENS=${1:-2048}

python -m vllm.entrypoints.openai.api_server \
  --model "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen-4b" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port 8505 \
  --max-num-batched-tokens $MAX_TOKENS
