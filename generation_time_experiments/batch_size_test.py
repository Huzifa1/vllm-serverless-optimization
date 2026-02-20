import time
import statistics
from vllm import LLM, SamplingParams

MODEL_NAME = "../models/qwen-4b"
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NUM_TRIALS = 1

# Target and background prompts are identical on purpose
# This guarantees:
#     Same prefill cost
#     Same decode length
#     Same completion time
PROMPT = "hi" * 500
MAX_TOKENS = 1
TEMPERATURE = 0.0

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)


results = {}

for batch_size in BATCH_SIZES:
    latencies = []

    prompts = [PROMPT] * batch_size

    # Warmup
    llm.generate(prompts, sampling_params)

    for _ in range(NUM_TRIALS):
        start = time.perf_counter()

        outputs = llm.generate(prompts, sampling_params)

        end = time.perf_counter()
        latency = end - start
        latencies.append(latency)

    if NUM_TRIALS > 1:
        results[batch_size] = {
            "mean": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": statistics.quantiles(latencies, n=20)[18],
            "raw": latencies,
        }
    else:
        results[batch_size] = {
            "mean": latencies[0],
            "p50": latencies[0],
            "p95": latencies[0],
            "raw": latencies,
        }


print("\nBatch Size vs Target Request Latency\n")
for bs, stats in results.items():
    print(
        f"Batch {bs:>2}: "
        f"mean={stats['mean']:.3f}s "
        f"p50={stats['p50']:.3f}s "
        f"p95={stats['p95']:.3f}s"
    )

