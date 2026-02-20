import asyncio
import time
import aiohttp
import sys

URL = "http://localhost:8505/v1/completions"
MODEL = "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models/qwen-4b"


def make_prompt(num_tokens: int) -> str:
    return str(num_tokens) + " " + "hello " * num_tokens

async def send_solo_request_async(prompt: str, max_tokens: int):
    """
    Sends a single request asynchronously.
    Returns (latency_seconds, response_json).
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }

    start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Request failed: {text}")

            result = await resp.json()

    duration = time.perf_counter() - start
    return duration, result

async def run_concurrent_solo_requests(inputs, prompts):
    
    tasks = [
        send_solo_request_async(prompts[i], inputs[i][1])
        for i in range(len(inputs))
    ]

    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start

    print(f"\nTotal wall-clock time: {total_time:.3f}s")

    for i, (duration, result) in enumerate(results):
        text_len = len(result["choices"][0]["text"].split())
        print(f"Request {i} solo time (async): {duration:.3f}s")
        print(f"Request {i}: output words ≈ {text_len}\n")

async def run_solo(prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False
    }
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Request failed: {text}")
            data = await resp.json()
    
    end = time.time()
    return end - start, data

if __name__ == "__main__":
    test_type = sys.argv[1]
    
    inputs = []
    if test_type == "mix":
        inputs = [
            (4096, 64),
            (2058, 512),
            (2058, 8),
            (512, 1024),
            (128, 32),
            (8192, 64),
            (16384, 128),
            (16384, 4096),
            (32750, 1),
            (30000, 2048),
        ]
    elif test_type == "prefill":
        inputs = [
            (128, 1),
            (256, 1),
            (512, 1),
            (1024, 1),
            (2048, 1),
            (4096, 1),
            (8192, 1),
            (16384, 1),
            (30000, 1),
        ]
    elif test_type == "decode":
        inputs = [
            (1, 128),
            (1, 256),
            (1, 512),
            (1, 1024),
            (1, 2048),
            (1, 4096),
            (1, 8192),
            (1, 16384),
            (1, 30000),
        ]
    
    prompts = [make_prompt(input[0]) for input in inputs]
    
    start = time.time()
    for i, input in enumerate(inputs):
        p = prompts[i]
        mt = input[1]
        duration, result = asyncio.run(run_solo(p, mt))
        text_len = len(result["choices"][0]["text"].split())
        print(f"Request {i} solo time: {duration:.3f}s")
        print(f"Request {i}: output words ≈ {text_len}\n")
    end = time.time()
    print(f"Total solo time: {end - start:.3f} seconds")
        
    print("\n\n\n")
    
    asyncio.run(run_concurrent_solo_requests(inputs, prompts))
    
    print("\n\n\n")