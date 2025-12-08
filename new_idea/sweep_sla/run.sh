export CUDA_VISIBLE_DEVICES="0"

MODELS_DIR=../../models
MODEL_NAME="${MODELS_DIR}/llama3-3b"

vllm bench sweep serve_sla \
 --serve-cmd "vllm serve $MODEL_NAME" \
 --bench-cmd "vllm bench serve --model $MODEL_NAME --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path ../ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10" \
 --sla-params sla_hparams.json \
 --sla-variable max_concurrency \
 -o results