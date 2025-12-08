import shutil
import sys
from pathlib import Path

def apply_vllm_changes(vllm_path):
    vllm_dir = Path(vllm_path)
    
    targets = [
        vllm_dir / "benchmarks/serve.py",
        vllm_dir / "benchmarks/datasets.py",
        vllm_dir / "benchmarks/lib/endpoint_request_func.py",
        vllm_dir / "benchmarks/lib/custom_scheduler.py"
    ]
    
    for target in targets:
        shutil.copy2(f"vllm_files/{target.name}", target)
    
    print(f"Changes applied!")

if __name__ == "__main__":
    try:
        import vllm
        import os
        vllm_path = os.path.dirname(vllm.__file__)
    except ImportError:
        print("Error: 'vllm' is not installed.")
        sys.exit(1)
        
    apply_vllm_changes(vllm_path)