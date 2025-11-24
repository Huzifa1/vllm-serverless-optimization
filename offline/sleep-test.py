from vllm import LLM
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODEL_PATH = "../models/llama3-3b"

# Initialize the LLM with sleep mode enabled
print("Initializing model with sleep mode enabled...")
start = time.perf_counter()
llm = LLM(MODEL_PATH, enable_sleep_mode=True)
end = time.perf_counter()
print(f"Model initialization took {end - start:.2f} seconds")

# Sleep level 1
# Put the engine to sleep (level=1: offload weights to CPU RAM, discard KV cache)
print("Putting model to sleep (level 1)...")
start = time.perf_counter()
llm.sleep(level=1)
end = time.perf_counter()
print(f"Sleep level 1 took {end - start:.2f} seconds")

# Wake up the engine (restore weights)
print("Waking up model from sleep (level 1)...")
start = time.perf_counter()
llm.wake_up()
end = time.perf_counter()
print(f"Wake up from sleep level 1 took {end - start:.2f} seconds")

# Sleep level 2
# Put the engine to sleep (level=2: discard both weights and KV cache)
print("Putting model to sleep (level 2)...")
start = time.perf_counter()
llm.sleep(level=2)
end = time.perf_counter()
print(f"Sleep level 2 took {end - start:.2f} seconds")

# Reallocate weights memory only
print("Reallocating weights memory...")
start = time.perf_counter()
llm.wake_up()
end = time.perf_counter()
print(f"Reallocate weights took {end - start:.2f} seconds")

# Load weights in-place
print("Reloading weights in-place...")
start = time.perf_counter()
llm.collective_rpc("reload_weights")
end = time.perf_counter()
print(f"Reload weights took {end - start:.2f} seconds")

# Reset KV cache
print("Reset prefix cache...")
start = time.perf_counter()
llm.reset_prefix_cache()
end = time.perf_counter()
print(f"Reset prefix cache took {end - start:.2f} seconds")