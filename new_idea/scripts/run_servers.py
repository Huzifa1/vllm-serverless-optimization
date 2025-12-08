import asyncio
import subprocess
import time
from typing import Optional
import aiohttp
import sys
import os
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json

def extract_port(url: str) -> int | None:
    """
    Extract the port number from a URL.
    Returns:
        - The explicit port number, if present.
        - The default port for the scheme (80 for http, 443 for https) if no port is specified.
        - None if port cannot be determined (e.g., unknown scheme).
    """
    parsed = urlparse(url)

    # If explicit port exists
    if parsed.port:
        return parsed.port

    # No explicit port: return default based on scheme
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443

    # Unknown scheme with no port
    return None

def run_vllm_server(model_name: str, port: int, log_file="./server.log"):
    """
    Start a vLLM OpenAI-compatible API server with sleep mode enabled.
    """

    # MAIN_DIR = repo_root = directory two levels above this script
    script_path = Path(__file__).resolve()
    MAIN_DIR = script_path.parent.parent.parent  # ../../../ relative to this file
    MODELS_DIR = MAIN_DIR / "models"
    LOGS_DIR = script_path.parent.parent / "logs"
    
    os.makedirs(LOGS_DIR, exist_ok=True)  # create logs dir if missing

    # Environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    env["VLLM_LOGGING_CONFIG_PATH"] = str(MAIN_DIR / "vllm_logging_config.json")
    env["VLLM_SERVER_DEV_MODE"] = "1"  # Exposes dev endpoints

    # Command to run
    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODELS_DIR / model_name),
        "--port", str(port),
        "--enable-sleep-mode"
    ]

    # Open log file for stdout+stderr
    with open(f"{LOGS_DIR}/{log_file}", "w") as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT  # redirects stderr to same log
        )

    return process


async def post(url: str, session: aiohttp.ClientSession, data=None):
        try:
            if data is None:
                async with session.post(url) as r:
                    await r.read()
                    return r.status
            else:
                async with session.post(url, json=data) as r:
                    await r.read()
                    return r.status
        except Exception as e:
            raise ValueError(f"Error sending POST to {url}: {e}")

async def start_server(base_url:str, model_name: str, check_every: float, timeout: int, session: aiohttp.ClientSession) -> int:
    """
    Start vLLM server with sleep mode enabled,
    wait until it's ready (max 60s),
    and return the PID.
    """
    
    port = extract_port(base_url)
    if port is None:
        raise ValueError(f"The provided url {base_url} does not have a valid port")
    
    name = model_name
    if "/" in model_name:
        name = model_name.split("/")[-1]
    log_file = f"server_{name}.log"
    
    # Start the server as a background process
    proc = run_vllm_server(
        model_name=model_name,
        port=port,
        log_file=log_file
    )
    server_pid = proc.pid
    print(f"Started server with PID {server_pid}", flush=True)

    # Wait for the server to be ready
    print(f"Waiting for vLLM server to be ready (max {timeout}s)...", flush=True)
    start_time = time.time()
    while True:
        try:
            async with session.get(f"{base_url}/is_sleeping") as r:
                if r.status == 200:
                    data = await r.json()
                    is_sleeping = data.get("is_sleeping", False)
                    
                    if not is_sleeping:
                        end_time = time.time()
                        print(f"Server is ready after {end_time - start_time:.3f} seconds.", flush=True)
                        return server_pid
        except:
            # Server is not up yet
            pass
    
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            print(f"Timed out waiting for vLLM after {timeout} seconds.", flush=True)
            # kill the server before exiting
            try:
                os.killpg(os.getpgid(server_pid), 9)
            except:
                pass
            sys.exit(1)

        await asyncio.sleep(check_every)

async def sleep_server(base_url: str, level: int, check_every: float, server_pid: int, timeout: int, session: aiohttp.ClientSession):    
    start = time.time()
    
    print(f"Executing sleep level {level}", flush=True)
    await post(f"{base_url}/sleep?level={level}", session)
    
    # Wait until action is complete
    start_time = time.time()
    while True:
        async with session.get(f"{base_url}/is_sleeping") as r:
            if r.status == 200:
                data = await r.json()
                is_sleeping = data.get("is_sleeping", False)
                if is_sleeping:
                    end_time = time.time()
                    print(f"Server is sleeping after {end_time - start:.3f} seconds.", flush=True)
                    break

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            print(f"Timed out waiting for vLLM to sleep after {timeout} seconds.", flush=True)
            # kill the server before exiting
            try:
                os.killpg(os.getpgid(server_pid), 9)
            except:
                pass
            sys.exit(1)  

        await asyncio.sleep(check_every)
    
def get_gpu_memory_for_pid() -> Optional[int]:
    """
    Return GPU memory (MiB) used by the given pid, or None if it cannot be determined.
    This uses `nvidia-smi --query-compute-apps=pid,used_memory` and sums matches.
    """
    try:
        # Query compute apps (running CUDA processes).
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=5
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        # nvidia-smi not available or failed
        return None

    total = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # format: "<pid>, <used_memory>"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            used_mem = int(parts[1])  # MiB
            total += used_mem
        except ValueError:
            continue

    return total

    
async def run_servers(model_to_port_map_path: str, base_url: str, check_every: float, sleep_level: int, timeout: int):
    # Create session
    connector = aiohttp.TCPConnector(
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=("https://" in base_url),
    )

    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=True,
        timeout=aiohttp.ClientTimeout(total=6 * 60 * 60),
    ) as session:
        # Parse model_to_port_map
        model_to_port_map = {}
        try: 
            with open(model_to_port_map_path, "r") as f:
                model_to_port_map = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{model_to_port_map_path}' does not exist!")
        except json.JSONDecodeError:
            raise ValueError(f"The file '{model_to_port_map_path}' is not valid JSON!")
            
        # Get current process pid and measure GPU memory before launching
        gpu_before = get_gpu_memory_for_pid()
        if gpu_before is None:
            print("Warning: could not determine GPU memory before launching (nvidia-smi missing or query failed).", flush=True)
        
        start_time = time.time()
        server_pids = []
        
        for model,port in model_to_port_map.items():
            url = f"{base_url}:{port}"
            server_pid = await start_server(base_url=url, model_name=model, check_every=check_every, timeout=timeout, session=session)
            server_pids.append(server_pid)
            await sleep_server(base_url=url, level=sleep_level, check_every=check_every, server_pid=server_pid, timeout=timeout, session=session)
            
        end_time = time.time()
        total_time = end_time - start_time
        
        # Measure GPU memory after launching
        gpu_after = get_gpu_memory_for_pid()
        if gpu_after is None:
            print("Warning: could not determine GPU memory after launching (nvidia-smi missing or query failed).", flush=True)
        
        # Print summary
        print("-" * 40, flush=True)
        print(f"Total time spent launching & sleeping servers: {total_time:.3f} seconds", flush=True)
        print(f"Server IDs (PIDs): {server_pids}", flush=True)

        if gpu_before is not None and gpu_after is not None:
            delta = gpu_after - gpu_before
            print(f"GPU memory used by this process: {delta} MiB", flush=True)
        else:
            print("GPU memory delta: unknown (measurement unavailable).", flush=True)

        print("-" * 40, flush=True)

async def main():
    parser = argparse.ArgumentParser(description="Run vLLM servers")
    parser.add_argument(
        "--model-to-port-map-file",
        type=str,
        help="Path to JSON file containing model to port mappings",
        required=True
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL to serve the servers (without port)",
        default="http://127.0.0.1"
    )
    parser.add_argument(
        "--check-every",
        type=float,
        help="Decides how often to check if the a server has reached a certain state",
        default=1
    )
    parser.add_argument(
        "--sleep-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="Specify sleep level",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Specify time to wait for server init in seconds",
    )
    
    args = parser.parse_args()

    await run_servers(args.model_to_port_map_file, args.base_url, args.check_every, args.sleep_level, args.timeout)
    
if __name__ == "__main__":
    asyncio.run(main())