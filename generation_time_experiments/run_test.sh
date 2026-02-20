for batch_size in 1 2 4 8 16 32 64
do
    bash bench_latency.sh qwen-4b logs/batch_size/$batch_size.log $batch_size
done