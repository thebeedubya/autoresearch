"""
Autoresearch Phase 1b: Model Ladder
Benchmark progressively larger models to find the quality/speed sweet spot.
Uses optimal parameters discovered from Phase 1a sweep.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import (
    ollama_api, unload_model, run_single_benchmark,
    get_gpu_memory, BENCH_PROMPTS, RESULTS_DIR, print_summary,
)

QUALITY_PROMPT = {
    "name": "quality_eval",
    "prompt": """You are a JSON-only classifier. Given the following social media post, classify it.

Post: "Just shipped our new ML pipeline - 3x faster inference with quantized models on consumer GPUs. Open source coming next week!"

Return ONLY this JSON, no other text:
{"category": "tech", "sentiment": "positive", "engagement_score": 0.0, "topics": [], "action": "none"}

Rules:
- category: one of [tech, cannabis, business, personal, news]
- sentiment: one of [positive, negative, neutral]
- engagement_score: 0.0-1.0 float
- topics: max 3 keyword strings
- action: one of [none, reply, repost, save]""",
    "num_predict": 128,
}


def check_model_available(model):
    """Check if model is pulled."""
    result = ollama_api("/api/tags")
    if result:
        for m in result.get("models", []):
            if m["name"].startswith(model.split(":")[0]) and (
                ":" not in model or m["name"] == model
            ):
                size_gb = m.get("size", 0) / (1024**3)
                print(f"  Found: {m['name']} ({size_gb:.1f} GB)")
                return True
    return False


def benchmark_model(model, options=None):
    """Full benchmark of a single model."""
    if options is None:
        options = {"num_ctx": 4096}  # conservative default

    print(f"\n{'#'*60}")
    print(f"# MODEL: {model}")
    print(f"{'#'*60}")

    # Unload any existing model
    unload_model(model)

    # Load model
    print("  Loading model...")
    start = time.perf_counter()
    warmup = ollama_api("/api/generate", {
        "model": model,
        "prompt": "Hello",
        "stream": False,
        "think": False,
        "options": {**options, "num_predict": 1},
    }, timeout=300)
    load_time = time.perf_counter() - start

    if not warmup:
        print(f"  FAILED to load {model}")
        return None

    print(f"  Loaded in {load_time:.1f}s")

    # Get memory footprint
    mem = get_gpu_memory()
    ps = ollama_api("/api/ps")
    if ps and ps.get("models"):
        m = ps["models"][0]
        print(f"  Processor: {m.get('details', {}).get('quantization_level', 'unknown')}")
        print(f"  Size: {m.get('size', 0) / (1024**3):.1f} GB")
        vram = m.get("size_vram", 0) / (1024**3)
        print(f"  VRAM: {vram:.1f} GB")

    # Standard benchmarks
    results = []
    all_prompts = BENCH_PROMPTS + [QUALITY_PROMPT]
    for prompt in all_prompts:
        print(f"\n  [{prompt['name']}] ({prompt['num_predict']} tokens)...")
        runs = []
        for i in range(2):
            r = run_single_benchmark(model, prompt, options)
            if r:
                runs.append(r)
                print(f"    run {i+1}: {r['tokens_per_sec']} tok/s (prompt: {r['prompt_tokens_per_sec']} tok/s)")

        if runs:
            avg_gen = sum(r["tokens_per_sec"] for r in runs) / len(runs)
            avg_prompt = sum(r["prompt_tokens_per_sec"] for r in runs) / len(runs)
            results.append({
                "model": model,
                "prompt_name": prompt["name"],
                "avg_gen_tps": round(avg_gen, 1),
                "avg_prompt_tps": round(avg_prompt, 1),
                "runs": runs,
                "memory": mem,
                "load_time_s": round(load_time, 1),
            })

    # Quality check — parse the JSON output
    print(f"\n  Quality check (JSON classifier)...")
    quality_result = ollama_api("/api/generate", {
        "model": model,
        "prompt": QUALITY_PROMPT["prompt"],
        "stream": False,
        "think": False,
        "options": {**options, "num_predict": 128},
    }, timeout=120)
    if quality_result:
        response = quality_result.get("response", "")
        print(f"  Raw output: {response[:200]}")
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
                valid_keys = {"category", "sentiment", "engagement_score", "topics", "action"}
                has_keys = set(parsed.keys()) & valid_keys
                print(f"  JSON valid: YES ({len(has_keys)}/{len(valid_keys)} expected keys)")
                print(f"  Parsed: {json.dumps(parsed, indent=2)}")
            else:
                print(f"  JSON valid: NO (no JSON found in output)")
        except json.JSONDecodeError:
            print(f"  JSON valid: NO (parse error)")

    return results


def run_model_ladder():
    """Run benchmarks across all available models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models = [
        "qwen3:8b",
        "qwen3:14b",
        "qwen3:32b",
        "qwen3:30b",   # MoE variant
        "qwen3:235b",  # The big one
    ]

    # Check which models are available
    print("Checking available models...")
    available = []
    for m in models:
        if check_model_available(m):
            available.append(m)
        else:
            print(f"  {m}: not pulled (skipping)")

    if not available:
        print("No models available!")
        return

    print(f"\nWill benchmark: {', '.join(available)}")

    all_results = {}
    for model in available:
        results = benchmark_model(model)
        if results:
            all_results[model] = results

    # Save all results
    results_file = RESULTS_DIR / f"model_ladder_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Summary table
    print(f"\n{'='*80}")
    print("MODEL LADDER RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Prompt':<15} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Load (s)':>10}")
    print("-" * 70)
    for model, results in all_results.items():
        for r in results:
            print(f"{model:<20} {r['prompt_name']:<15} {r['avg_gen_tps']:>10.1f} {r['avg_prompt_tps']:>12.1f} {r['load_time_s']:>10.1f}")

    # Overall model ranking by avg generation speed
    print(f"\n{'='*80}")
    print("OVERALL MODEL RANKING")
    print(f"{'='*80}")
    model_avgs = {}
    for model, results in all_results.items():
        speeds = [r["avg_gen_tps"] for r in results]
        model_avgs[model] = sum(speeds) / len(speeds) if speeds else 0

    for i, (model, avg) in enumerate(sorted(model_avgs.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  #{i}: {model:<20} {avg:.1f} tok/s")


if __name__ == "__main__":
    print(f"Autoresearch Model Ladder v1.0")
    print(f"GPU: AMD Radeon 8060S (Strix Halo gfx1151, 96 GiB UMA)")
    print(f"Time: {datetime.now().isoformat()}")
    run_model_ladder()
