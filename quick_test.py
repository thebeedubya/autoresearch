"""Quick validation — run 3 configs with 1 prompt to verify the loop works."""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from benchmark import (
    ollama_api, unload_model, benchmark_config, print_summary,
    BENCH_PROMPTS, RESULTS_DIR
)
import json
from datetime import datetime

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"

# Override to just use short prompt for speed
BENCH_PROMPTS_QUICK = [BENCH_PROMPTS[0]]  # short_gen only

import benchmark
benchmark.BENCH_PROMPTS = BENCH_PROMPTS_QUICK

configs = [
    ("baseline", {}),
    ("ctx_2048", {"num_ctx": 2048}),
    ("ctx_4096_batch_1024", {"num_ctx": 4096, "num_batch": 1024}),
]

print(f"Quick validation — {len(configs)} configs x 1 prompt x 2 runs")
print(f"Model: {model}\n")

all_results = []
for name, opts in configs:
    results = benchmark_config(model, opts, name, runs=2)
    if results:
        all_results.extend(results)

print_summary(all_results)
