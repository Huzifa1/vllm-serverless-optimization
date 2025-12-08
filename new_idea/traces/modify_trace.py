import random
import jsonlines
import sys
import argparse
import os
from transformers import AutoConfig


MODELS_DIR = "/local/huzaifa/workspace/vLLM/vllm-serverless-optimization/models"

model_max_lengths = {
    "llama3-3b": 0,
    "qwen-1.8b": 0,
    "qwen2.5-3b": 0,
}


def get_model_max_len(model_path: str) -> int:
    """
    Returns the maximum input length for a HuggingFace model **without** loading weights.
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # The attribute name differs by architecture:
    candidates = [
        "max_position_embeddings",
        "max_seq_len",
        "n_positions",
        "seq_length",
        "model_max_length",
    ]

    for attr in candidates:
        if hasattr(cfg, attr):
            value = getattr(cfg, attr)
            if isinstance(value, int) and value > 0:
                return value

    # Fallback: try tokenizer (often defines model_max_length)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tok, "model_max_length") and tok.model_max_length > 0:
            return tok.model_max_length
    except Exception:
        pass

    raise ValueError(
        f"Could not determine max input length from {model_path}. "
        f"Inspect its config.json manually."
    )

def make_prompt(doc_id: int, input_length: int) -> str:
    # prefix with doc id to avoid prefix cache and then 'hi' repeated
    return str(doc_id) + " " + " ".join(["hi"] * input_length)


def main(args):
    random.seed(args.seed)

    input_path = "conversation_trace.jsonl"
    output_path = f"modified_conversation_trace_{args.num_prompts}.jsonl"

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    new_lines = []
    processed = 0
    skipped_count = 0
    truncated_count = 0
    
    for model in model_max_lengths.keys():
        model_limit = get_model_max_len(f"{MODELS_DIR}/{model}")
        model_max_lengths[model] = model_limit
            

    with jsonlines.open(input_path) as reader:
        # iterate until we collect desired number or EOF
        for i, line in enumerate(reader):
            if len(new_lines) >= args.num_prompts:
                break

            orig_input_length = int(line["input_length"])

            # Pick random model
            model_name = random.choice(list(model_max_lengths.keys()))
            model_limit = model_max_lengths.get(model_name, 10**9)

            if orig_input_length > model_limit:
                if args.mode == "truncate":
                    # Truncate input length to model limit
                    new_input_length = model_limit
                    truncated_count += 1
                    # update line
                    line["input_length"] = new_input_length
                    line["prompt"] = make_prompt(len(new_lines), new_input_length)
                    line["model_name"] = f"{MODELS_DIR}/{model_name}"
                    new_lines.append(line)
                elif args.mode == "skip":
                    skipped_count += 1
                    # skip this row and continue
                    continue
            else:
                # within model limit — keep row, generate prompt
                line["prompt"] = make_prompt(len(new_lines), orig_input_length)
                line["model_name"] = f"{MODELS_DIR}/{model_name}"
                new_lines.append(line)

            processed += 1

    # After reading file:
    if len(new_lines) < args.num_prompts:
        print(
            f"Warning: reached end of input file but only produced {len(new_lines)} rows "
            f"(requested {args.num_prompts}).",
            file=sys.stderr,
        )

    # Write modified trace
    with jsonlines.open(output_path, mode="w") as writer:
        writer.write_all(new_lines)

    # Summary to stderr
    print(f"Produced: {len(new_lines)} rows -> {output_path}")
    print(f"Processed rows considered: {processed}")
    print(f"Skipped rows: {skipped_count}")
    print(f"Truncated rows (in truncate mode): {truncated_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare modified conversation trace with model_name and prompt. "
        "If a prompt's input_length exceeds a model's max, either truncate or skip."
    )
    parser.add_argument("--num_prompts", type=int, help="Number of output prompts to produce")
    parser.add_argument(
        "--mode",
        choices=["truncate", "skip"],
        default="truncate",
        help="Behavior when input_length > model max: 'truncate' or 'skip' (default: truncate)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model selection (default: 42)",
    )

    args = parser.parse_args()
    main(args)
