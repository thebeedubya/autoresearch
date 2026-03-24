"""
Autoresearch Inference Optimizer — Phase 1
Karpathy-style iterative optimization loop for Ollama on AMD ROCm.

Iterates over inference parameters, benchmarks each config,
logs results, and converges on the optimal configuration.
"""

import json
import time
import subprocess
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

OLLAMA_API = "http://localhost:11434"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Standard benchmark prompts (varying complexity)
BENCH_PROMPTS = [
    {
        "name": "short_gen",
        "prompt": "Write a Python function that checks if a number is prime.",
        "num_predict": 128,
    },
    {
        "name": "medium_gen",
        "prompt": "Explain the difference between TCP and UDP. Include code examples in Python for both.",
        "num_predict": 512,
    },
    {
        "name": "long_gen",
        "prompt": "Write a comprehensive REST API in Python using FastAPI that manages a todo list with CRUD operations, authentication, and database integration. Include all imports and complete implementation.",
        "num_predict": 1024,
    },
]


def ollama_api(endpoint, data=None, timeout=300):
    """Call Ollama API."""
    url = f"{OLLAMA_API}{endpoint}"
    if data:
        req = Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    else:
        req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"  ERROR: Ollama API call failed: {e}")
        return None


def unload_model(model):
    """Unload model from memory to reset state."""
    ollama_api("/api/generate", {"model": model, "keep_alive": 0})
    time.sleep(2)


def get_gpu_memory():
    """Get GPU memory usage from Ollama ps."""
    result = ollama_api("/api/ps")
    if result and result.get("models"):
        m = result["models"][0]
        return {
            "size_vram": m.get("size_vram", 0),
            "size": m.get("size", 0),
        }
    return {}


def run_single_benchmark(model, prompt_config, options, think=False):
    """Run a single inference benchmark and return metrics."""
    data = {
        "model": model,
        "prompt": prompt_config["prompt"],
        "stream": False,
        "options": {**options, "num_predict": prompt_config["num_predict"]},
    }
    if not think:
        data["think"] = False

    start = time.perf_counter()
    result = ollama_api("/api/generate", data, timeout=600)
    wall_time = time.perf_counter() - start

    if not result:
        return None

    eval_count = result.get("eval_count", 0)
    eval_duration_s = result.get("eval_duration", 1) / 1e9
    prompt_eval_count = result.get("prompt_eval_count", 0)
    prompt_eval_duration_s = result.get("prompt_eval_duration", 1) / 1e9

    return {
        "prompt_name": prompt_config["name"],
        "num_predict": prompt_config["num_predict"],
        "eval_count": eval_count,
        "eval_duration_s": round(eval_duration_s, 3),
        "tokens_per_sec": round(eval_count / eval_duration_s, 1) if eval_duration_s > 0 else 0,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_s": round(prompt_eval_duration_s, 3),
        "prompt_tokens_per_sec": round(prompt_eval_count / prompt_eval_duration_s, 1) if prompt_eval_duration_s > 0 else 0,
        "wall_time_s": round(wall_time, 3),
        "total_duration_s": round(result.get("total_duration", 0) / 1e9, 3),
        "load_duration_s": round(result.get("load_duration", 0) / 1e9, 3),
    }


def benchmark_config(model, options, config_name, runs=2):
    """Benchmark a specific configuration across all prompts."""
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Options: {json.dumps(options, indent=2)}")
    print(f"{'='*60}")

    # Unload and reload with new config
    unload_model(model)

    # Warm up — first inference loads the model
    print("  Warming up...")
    warmup = ollama_api("/api/generate", {
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "think": False,
        "options": {**options, "num_predict": 1},
    }, timeout=120)
    if not warmup:
        print("  FAILED to load model with this config")
        return None

    # Get memory usage after load
    mem = get_gpu_memory()

    results = []
    for prompt_config in BENCH_PROMPTS:
        prompt_results = []
        for run in range(runs):
            print(f"  [{prompt_config['name']}] run {run+1}/{runs}...", end=" ", flush=True)
            r = run_single_benchmark(model, prompt_config, options)
            if r:
                prompt_results.append(r)
                print(f"{r['tokens_per_sec']} tok/s")
            else:
                print("FAILED")

        if prompt_results:
            avg_tps = sum(r["tokens_per_sec"] for r in prompt_results) / len(prompt_results)
            avg_prompt_tps = sum(r["prompt_tokens_per_sec"] for r in prompt_results) / len(prompt_results)
            results.append({
                "config_name": config_name,
                "prompt_name": prompt_config["name"],
                "num_predict": prompt_config["num_predict"],
                "avg_tokens_per_sec": round(avg_tps, 1),
                "avg_prompt_tokens_per_sec": round(avg_prompt_tps, 1),
                "runs": prompt_results,
                "memory": mem,
                "options": options,
            })

    return results


def phase1_parameter_sweep(model):
    """Phase 1: Sweep key Ollama parameters to find optimal config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    # Parameter space to explore
    configs = [
        # Baseline
        ("baseline", {}),

        # Context size variations
        ("ctx_2048", {"num_ctx": 2048}),
        ("ctx_4096", {"num_ctx": 4096}),
        ("ctx_8192", {"num_ctx": 8192}),
        ("ctx_16384", {"num_ctx": 16384}),

        # Batch size variations
        ("batch_128", {"num_batch": 128}),
        ("batch_256", {"num_batch": 256}),
        ("batch_1024", {"num_batch": 1024}),
        ("batch_2048", {"num_batch": 2048}),

        # Thread count variations
        ("threads_8", {"num_thread": 8}),
        ("threads_12", {"num_thread": 12}),
        ("threads_16", {"num_thread": 16}),
        ("threads_24", {"num_thread": 24}),

        # Combined best guesses
        ("combo_small_ctx_big_batch", {"num_ctx": 4096, "num_batch": 1024}),
        ("combo_min_ctx", {"num_ctx": 2048, "num_batch": 512}),

        # Flash attention (if supported)
        ("flash_attn_on", {"flash_attention": True}),
        ("flash_attn_off", {"flash_attention": False}),
    ]

    print(f"\nAutoresearch Phase 1: Parameter Sweep")
    print(f"Model: {model}")
    print(f"Configs to test: {len(configs)}")
    print(f"Prompts per config: {len(BENCH_PROMPTS)}")
    print(f"Runs per prompt: 2")
    print(f"Estimated time: ~{len(configs) * 3 * 2 * 15 // 60} minutes\n")

    for config_name, options in configs:
        try:
            results = benchmark_config(model, options, config_name)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"  ERROR in config {config_name}: {e}")

    # Save raw results
    results_file = RESULTS_DIR / f"sweep_{model.replace(':', '_')}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to: {results_file}")

    # Generate summary
    print_summary(all_results)
    save_summary_csv(all_results, RESULTS_DIR / f"sweep_{model.replace(':', '_')}_{timestamp}.csv")

    return all_results


def print_summary(results):
    """Print a ranked summary of all configs."""
    print(f"\n{'='*80}")
    print("AUTORESEARCH RESULTS — Ranked by avg generation tok/s")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'Prompt':<15} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'VRAM':>10}")
    print("-" * 80)

    # Sort by generation speed
    sorted_results = sorted(results, key=lambda x: x["avg_tokens_per_sec"], reverse=True)

    for r in sorted_results:
        vram_gb = r.get("memory", {}).get("size_vram", 0) / (1024**3)
        print(f"{r['config_name']:<35} {r['prompt_name']:<15} {r['avg_tokens_per_sec']:>10.1f} {r['avg_prompt_tokens_per_sec']:>12.1f} {vram_gb:>9.1f}G")

    # Find best overall config
    config_avg = {}
    for r in results:
        name = r["config_name"]
        if name not in config_avg:
            config_avg[name] = []
        config_avg[name].append(r["avg_tokens_per_sec"])

    print(f"\n{'='*80}")
    print("OVERALL RANKING (avg across all prompts)")
    print(f"{'='*80}")
    ranked = sorted(config_avg.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
    for i, (name, scores) in enumerate(ranked, 1):
        avg = sum(scores) / len(scores)
        print(f"  #{i}: {name:<35} {avg:.1f} tok/s")

    if ranked:
        print(f"\n  WINNER: {ranked[0][0]} at {sum(ranked[0][1])/len(ranked[0][1]):.1f} tok/s")


def save_summary_csv(results, path):
    """Save results as CSV for easy analysis."""
    if not results:
        return
    fields = ["config_name", "prompt_name", "num_predict", "avg_tokens_per_sec", "avg_prompt_tokens_per_sec"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV summary saved to: {path}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    print(f"Autoresearch Inference Optimizer v1.0")
    print(f"Target: {model}")
    print(f"GPU: AMD Radeon 8060S (Strix Halo gfx1151)")
    print(f"VRAM: 96 GiB UMA")
    print(f"Backend: Ollama ROCm")
    print(f"Time: {datetime.now().isoformat()}")

    phase1_parameter_sweep(model)
