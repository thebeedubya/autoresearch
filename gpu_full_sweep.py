"""
Autoresearch Phase 2: Full GPU Sweep
Now that ROCm is working (HSA_OVERRIDE_GFX_VERSION=11.0.0),
re-benchmark everything on GPU and run parameter optimization on 235B.
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
    save_summary_csv,
)
from model_ladder import QUALITY_PROMPT, benchmark_model

# Memory monitor
def check_memory():
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        free_gb = stat.ullAvailPhys / (1024**3)
        return free_gb
    except:
        return -1


def safe_benchmark(model, prompt_config, options, retries=2):
    """Run benchmark with memory safety checks."""
    free = check_memory()
    if free >= 0 and free < 1.0:
        print(f"  WARNING: Only {free:.1f} GB RAM free — skipping to avoid crash")
        return None

    for attempt in range(retries):
        result = run_single_benchmark(model, prompt_config, options)
        if result:
            return result
        if attempt < retries - 1:
            print(f"  Retrying ({attempt+2}/{retries})...")
            time.sleep(5)
    return None


def phase2_model_ladder_gpu():
    """Re-run model ladder with GPU acceleration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("PHASE 2A: MODEL LADDER ON GPU (ROCm)")
    print("=" * 70)

    # Check what's available
    result = ollama_api("/api/tags")
    if not result:
        print("ERROR: Can't reach Ollama")
        return

    available = {}
    for m in result.get("models", []):
        name = m["name"]
        size_gb = m.get("size", 0) / (1024**3)
        available[name] = size_gb
        print(f"  Available: {name} ({size_gb:.1f} GB)")

    # Models to benchmark (in order of size)
    test_models = []
    for pattern in ["qwen3:8b", "qwen3:32b", "usamakenway/Qwen3-235B-A22B-Q2_K-GGUF:latest"]:
        for name in available:
            if name == pattern or name.startswith(pattern.split(":")[0]):
                if name not in [t[0] for t in test_models]:
                    test_models.append((name, available[name]))
                    break

    print(f"\nWill benchmark {len(test_models)} models on GPU")

    all_results = {}
    for model_name, size_gb in test_models:
        print(f"\n{'#' * 60}")
        print(f"# {model_name} ({size_gb:.1f} GB)")
        print(f"{'#' * 60}")

        # Unload everything first
        for m in available:
            unload_model(m)
        time.sleep(3)

        results = benchmark_model(model_name, options={"num_ctx": 4096})
        if results:
            all_results[model_name] = results

    # Save results
    results_file = RESULTS_DIR / f"gpu_model_ladder_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("GPU MODEL LADDER RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Model':<50} {'Prompt':<15} {'Gen tok/s':>10} {'Prompt tok/s':>12}")
    print("-" * 90)
    for model_name, results in all_results.items():
        for r in results:
            print(f"{model_name:<50} {r['prompt_name']:<15} {r['avg_gen_tps']:>10.1f} {r['avg_prompt_tps']:>12.1f}")

    # Overall ranking
    print(f"\n{'=' * 80}")
    print("OVERALL GPU RANKING")
    print(f"{'=' * 80}")
    model_avgs = {}
    for model_name, results in all_results.items():
        speeds = [r["avg_gen_tps"] for r in results]
        model_avgs[model_name] = sum(speeds) / len(speeds) if speeds else 0

    for i, (model_name, avg) in enumerate(sorted(model_avgs.items(), key=lambda x: x[1], reverse=True), 1):
        print(f"  #{i}: {model_name:<50} {avg:.1f} tok/s")

    return all_results


def phase2_235b_optimization():
    """Phase 2B: Parameter sweep on 235B specifically."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = "usamakenway/Qwen3-235B-A22B-Q2_K-GGUF:latest"

    print(f"\n{'=' * 70}")
    print("PHASE 2B: 235B PARAMETER OPTIMIZATION")
    print(f"{'=' * 70}")

    # Key parameters for MoE on GPU
    # Focus on what matters most: context size, batch size, flash attention
    configs = [
        # Baseline
        ("baseline_4k", {"num_ctx": 4096}),

        # Context size — smaller = more VRAM for KV cache overhead
        ("ctx_2048", {"num_ctx": 2048}),
        ("ctx_8192", {"num_ctx": 8192}),
        ("ctx_16384", {"num_ctx": 16384}),
        ("ctx_32768", {"num_ctx": 32768}),

        # Batch size — affects prompt processing speed
        ("batch_256", {"num_ctx": 4096, "num_batch": 256}),
        ("batch_512", {"num_ctx": 4096, "num_batch": 512}),
        ("batch_1024", {"num_ctx": 4096, "num_batch": 1024}),
        ("batch_2048", {"num_ctx": 4096, "num_batch": 2048}),

        # Flash attention — critical for large models
        ("flash_on_4k", {"num_ctx": 4096, "flash_attention": True}),
        ("flash_on_8k", {"num_ctx": 8192, "flash_attention": True}),
        ("flash_on_16k", {"num_ctx": 16384, "flash_attention": True}),

        # Best combo candidates
        ("combo_flash_batch1024_4k", {"num_ctx": 4096, "num_batch": 1024, "flash_attention": True}),
        ("combo_flash_batch1024_8k", {"num_ctx": 8192, "num_batch": 1024, "flash_attention": True}),
        ("combo_2k_batch512_flash", {"num_ctx": 2048, "num_batch": 512, "flash_attention": True}),
    ]

    print(f"Model: {model}")
    print(f"Configs: {len(configs)}")
    print(f"Prompts: short_gen (128 tok) + medium_gen (512 tok)")
    print(f"Runs per prompt: 2")

    # Use shorter prompts for 235B sweep (long_gen takes too long per config)
    short_prompts = BENCH_PROMPTS[:2]  # short_gen + medium_gen only

    all_results = []

    for config_name, options in configs:
        print(f"\n{'=' * 60}")
        print(f"Config: {config_name}")
        print(f"Options: {json.dumps(options)}")
        print(f"{'=' * 60}")

        # Check memory
        free = check_memory()
        if free >= 0:
            print(f"  System RAM free: {free:.1f} GB")
            if free < 1.0:
                print(f"  SKIPPING — low memory")
                continue

        # Unload and reload
        unload_model(model)
        time.sleep(2)

        # Warm up
        print("  Loading model...")
        start = time.perf_counter()
        warmup = ollama_api("/api/generate", {
            "model": model,
            "prompt": "Hi",
            "stream": False,
            "think": False,
            "options": {**options, "num_predict": 1},
        }, timeout=300)
        load_time = time.perf_counter() - start

        if not warmup:
            print(f"  FAILED to load with config {config_name}")
            continue

        print(f"  Loaded in {load_time:.1f}s")

        # Get memory
        mem = get_gpu_memory()
        ps = ollama_api("/api/ps")
        if ps and ps.get("models"):
            m = ps["models"][0]
            vram = m.get("size_vram", 0) / (1024**3)
            total = m.get("size", 0) / (1024**3)
            print(f"  VRAM: {vram:.1f} GB / Total: {total:.1f} GB")

        # Run benchmarks
        for prompt_config in short_prompts:
            runs = []
            for i in range(2):
                print(f"  [{prompt_config['name']}] run {i+1}/2...", end=" ", flush=True)
                r = safe_benchmark(model, prompt_config, options)
                if r:
                    runs.append(r)
                    print(f"{r['tokens_per_sec']} tok/s (prompt: {r['prompt_tokens_per_sec']} tok/s)")
                else:
                    print("FAILED")

            if runs:
                avg_gen = sum(r["tokens_per_sec"] for r in runs) / len(runs)
                avg_prompt = sum(r["prompt_tokens_per_sec"] for r in runs) / len(runs)
                all_results.append({
                    "config_name": config_name,
                    "prompt_name": prompt_config["name"],
                    "avg_tokens_per_sec": round(avg_gen, 1),
                    "avg_prompt_tokens_per_sec": round(avg_prompt, 1),
                    "runs": runs,
                    "memory": mem,
                    "options": options,
                    "load_time_s": round(load_time, 1),
                })

    # Save results
    results_file = RESULTS_DIR / f"235b_optimization_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Print ranked summary
    print_summary(all_results)
    save_summary_csv(all_results, RESULTS_DIR / f"235b_optimization_{timestamp}.csv")

    return all_results


if __name__ == "__main__":
    print(f"Autoresearch Phase 2: Full GPU Sweep")
    print(f"GPU: AMD Radeon 8060S (Strix Halo gfx1151, 96 GiB UMA)")
    print(f"Backend: Ollama ROCm (HSA_OVERRIDE_GFX_VERSION=11.0.0)")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Phase 2A: Model ladder on GPU
    ladder_results = phase2_model_ladder_gpu()

    # Phase 2B: 235B optimization
    opt_results = phase2_235b_optimization()

    print("\n" + "=" * 70)
    print("AUTORESEARCH PHASE 2 COMPLETE")
    print("=" * 70)
