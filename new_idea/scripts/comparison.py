import argparse
import jsonlines

model_cold_starts = {
    "llama3-3b": 18,
    "qwen2.5-3b": 20,
    "qwen-1.8b": 17
}

def calc_cold_start(input_trace_file: str):
    total = 0
    previous_model = None
    current_model = None
    with jsonlines.open(input_trace_file, mode="r") as f:
        for line in f.iter():
            model = line["model_name"]
            model_name = model.split("/")[-1]
            current_model = model_name
            cold_start = model_cold_starts[model_name]
            if previous_model and current_model != previous_model:
                total += cold_start
            previous_model = current_model
            
    return total

def main():
    parser = argparse.ArgumentParser(description="Compare theoritical performance")
    parser.add_argument(
        "--input-trace-file",
        type=str,
        help="Path to JSONL file containing input trace",
        required=True
    )
    
    args = parser.parse_args()
    
    total_cold_start = calc_cold_start(args.input_trace_file)
    print(f"Total cold start is {total_cold_start} seconds")

if __name__ == "__main__":
    main()