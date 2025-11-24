import argparse
import requests

SERVER_URL = "http://localhost:"

def is_server_sleeping():
    """Check if the server is currently in sleep mode."""
    r = requests.get(f"{SERVER_URL}/is_sleeping")
    return r.json().get("is_sleeping", False)

def post(endpoint, data=None):
    """Helper for POST requests with optional JSON."""
    url = f"{SERVER_URL}/{endpoint}"
    try:
        if data is None:
            r = requests.post(url)
        else:
            r = requests.post(url, json=data)
        return r
    except Exception as e:
        print(f"Error sending POST to {url}: {e}")
        return None


def generate(prompt, model, max_tokens=128):
    """Call vLLM OpenAI-style /v1/completions endpoint."""
    
    url = f"{SERVER_URL}/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(url, json=payload)
        print(r.json()["choices"][0]["text"])
        return r.json()
    except Exception as e:
        print(r.json())
        print(f"Error during generation request: {e}")
        return None


def handle_sleep(level):
    """Execute sleep behavior based on sleep level."""
    print(f"Executing sleep level {level}")
    post(f"sleep?level={level}")


def handle_wakeup(level):
    """Execute wake-up behavior based on sleep level."""
    print(f"Executing wakeup for level {level}")

    # Wake up
    post("wake_up")

    if level == 2:
        # Reload weights
        post("collective_rpc", {"method": "reload_weights"})

        # Reset prefix cache
        post("reset_prefix_cache")


def handle_generate(prompt, model, max_tokens):
    """Execute a text generation request."""
    
    # Check if server is sleeping
    if is_server_sleeping():
        print("Server is currently sleeping. Cannot generate text.")
        return None
    
    print(f"Generating text: prompt='{prompt}', max_tokens={max_tokens}")
    return generate(prompt, model, max_tokens)


def main():
    parser = argparse.ArgumentParser(description="vLLM sleep/wakeup/generate client")

    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=["sleep", "wakeup", "generate"],
        help="Action to perform"
    )

    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Sleep level (1 or 2)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8500,
        help="Port of the vLLM server"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt for generate action"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Path to required model"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Max tokens for generate action"
    )

    args = parser.parse_args()

    # Validate level
    if args.level not in [1, 2]:
        raise ValueError("level must be 1 or 2")
    
    global SERVER_URL
    SERVER_URL += str(args.port)

    # Dispatch actions
    if args.action == "sleep":
        handle_sleep(args.level)

    elif args.action == "wakeup":
        handle_wakeup(args.level)

    elif args.action == "generate":
        if not args.prompt:
            raise ValueError("--prompt is required when action=generate")
        if not args.model:
            raise ValueError("--model is required when action=generate")
        handle_generate(args.prompt, args.model, args.max_tokens)
    


if __name__ == "__main__":
    main()
