# Try to reproduce "A4000 GPU Results" figure from 
# this blog: https://blog.vllm.ai/2025/10/26/sleep-mode.html

# Scenario: Running inference on Model A, switching to Model B,
# running inference on Model B, then repeating this pattern.
# (A→B→A→B→A→B)

# Compare 2 cases: Using sleep mode vs not using sleep mode.

import subprocess
import time
import requests
import sys
import os

shared_prompt = "Once upon a time in a land far, far away,"
max_tokens = 256
model_a = "../models/llama3-3b"
model_b = "../models/qwen2.5-3b"
port_a = 8500
port_b = 8501
sleep_level = 2


def start_server(sleep_enabled: str, model: str) -> int:
    """
    Start vLLM server with or without sleep mode enabled,
    wait until it's ready (max 60s),
    and return the PID.
    """
    
    model_name = model.split("/")[-1]
    log_path = f"server_{model_name}_sleep_{sleep_enabled}.log"
    port = port_a if model_name == "llama3-3b" else port_b
    
    # Start the server as a background process
    proc = subprocess.Popen(
        ["bash", "server.sh", sleep_enabled, model, log_path, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid   # so it runs independently like '&'
    )
    server_pid = proc.pid
    print(f"Started server with PID {server_pid}", flush=True)

    # Wait for the server to be ready
    print("Waiting for vLLM server to be ready (max 60s)...", flush=True)

    timeout = 60
    start_time = time.time()

    while True:
        try:
            r = requests.get(f"http://localhost:{port}/is_sleeping", timeout=1)
            if r.status_code == 200 and not r.json().get("is_sleeping", False):
                end_time = time.time()
                print(f"Server is ready after {end_time - start_time:.3f} seconds.", flush=True)
                return server_pid
        except requests.exceptions.RequestException:
            pass  # server not up yet

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            print(f"Timed out waiting for vLLM after {timeout} seconds.", flush=True)
            # kill the server before exiting
            try:
                os.killpg(os.getpgid(server_pid), 9)
            except:
                pass
            sys.exit(1)

        time.sleep(1)
        
def stop_server(server_pid: int):
    """Kill the vLLM server process given its PID."""
    print(f"Killing server with PID {server_pid}...", flush=True)
    try:
        start = time.time()
        os.killpg(os.getpgid(server_pid), 9)
        end = time.time()
        print(f"Server killed in {end - start:.3f} seconds.", flush=True)
    except Exception as e:
        print(f"Error killing server: {e}", flush=True)


def action_server(level: int, action: str, port: int):
    """Call client.py script."""
    print(f"Executing {action} server with level {level}...", flush=True)
    
    # Pass --action sleep, --level <level>
    start = time.time()
    subprocess.run([
        "python3",
        "client.py",
        "--action", f"{action}",
        "--port", f"{port}",
        "--level", f"{level}"
    ])
    
    # Wait until action is complete
    while True:
        r = requests.get(f"http://localhost:{port}/is_sleeping", timeout=1)
        if r.status_code == 200:
            is_sleeping = r.json().get("is_sleeping", False)
            if action == "sleep" and is_sleeping:
                end_time = time.time()
                print(f"Server is sleeping after {end_time - start:.3f} seconds.", flush=True)
                break
            
            if action == "wakeup" and not is_sleeping:
                end_time = time.time()
                print(f"Server is wakeup after {end_time - start:.3f} seconds.", flush=True)
                break
            
        time.sleep(1)
    
def generate_text(prompt: str, model:str, max_tokens: int = 128):
    """Call client.py script with action 'generate'."""
    print(f"Generating text for prompt: {prompt[:30]}...", flush=True)
    
    model_name = model.split("/")[-1]
    port = port_a if model_name == "llama3-3b" else port_b

    # Pass --action generate, --prompt <prompt>, --max_tokens <max_tokens>
    start = time.time()
    subprocess.run([
        "python3",
        "client.py",
        "--action", "generate",
        "--port", f"{port}",
        "--prompt", f"{prompt}",
        "--model", f"{model}",
        "--max_tokens", f"{max_tokens}"
    ])
    end = time.time()
    print(f"Text generated in {end - start:.3f} seconds.", flush=True)
    
def run_A_pipeline(model: str):
    
    model_name = model.split("/")[-1]
    sleep_enabled = "false"
    
    print(f"Load {model_name} with sleep_enabled={sleep_enabled}", flush=True)
    server_pid = start_server(sleep_enabled, model)
    print(f"Generate for {model_name}", flush=True)
    generate_text(shared_prompt, model, max_tokens)
    print(f"Kill {model_name}", flush=True)
    stop_server(server_pid)
    
    print("\n")
    print("*" * 50)
    print("\n")
    
def run_B_pipeline_1(model: str) -> int:
    
    model_name = model.split("/")[-1]
    sleep_enabled = "true"
    port = port_a if model_name == "llama3-3b" else port_b
    
    print(f"Load {model_name} with sleep_enabled={sleep_enabled}", flush=True)
    server_pid = start_server(sleep_enabled, model)
    print(f"Generate for {model_name}", flush=True)
    generate_text(shared_prompt, model, max_tokens)
    print(f"Put {model_name} to sleep", flush=True)
    action_server(level=sleep_level, action="sleep", port=port)
    
    print("\n")
    print("*" * 50)
    print("\n")
    
    return server_pid

def run_B_pipeline_2(model: str):
    model_name = model.split("/")[-1]
    port = port_a if model_name == "llama3-3b" else port_b
    
    print(f"Wake up {model_name}", flush=True)
    action_server(level=sleep_level, action="wakeup", port=port)
    print(f"Generate for {model_name}", flush=True)
    generate_text(shared_prompt, model, max_tokens)
    print(f"Put {model_name} to sleep", flush=True)
    action_server(level=sleep_level, action="sleep", port=port)
    
    print("\n")
    print("*" * 50)
    print("\n")

# Case A: Without sleep mode
print("=== Case A: Without sleep mode ===", flush=True)

run_A_pipeline(model_a)
run_A_pipeline(model_b)
run_A_pipeline(model_a)
run_A_pipeline(model_b)
run_A_pipeline(model_a)
run_A_pipeline(model_b)


# Case B: Without sleep mode
print("=== Case B: With sleep mode ===", flush=True)

# Start servers and send them to sleep
server_pid_a = run_B_pipeline_1(model_a)
server_pid_b = run_B_pipeline_1(model_b)

# Alternate waking up, generating, and sleeping
run_B_pipeline_2(model_a)
run_B_pipeline_2(model_b)
run_B_pipeline_2(model_a)
run_B_pipeline_2(model_b)
run_B_pipeline_2(model_a)
run_B_pipeline_2(model_b)

# Kill both servers at the end
stop_server(server_pid_a)
stop_server(server_pid_b)