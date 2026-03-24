"""
Autoresearch Phase 2B: Safe 235B Optimization
Only test configs that fit in the GPU/CPU split (66/95 layers).
Skip large context sizes that cause OOM.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark import (
    ollama_api, unload_model, run_single_benchmark,
    get_gpu_memory, BENCH_PROMPTS, RESULTS_DIR,
)

MODEL = "usamakenway/Qwen3-235B-A22B-Q2_K-GGUF:latest"


def check_memory():
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong),
            ]
        s = MS()
        s.dwLength = ctypes.sizeof(s)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return round(s.ullAvailPhys / (1024**3), 1)
    except:
        return -1


def run_config(config_name, options, prompts):
    """Run a single config safely."""
    print(f"\n{'=' * 60}")
    print(f"Config: {config_name}")
    print(f"Options: {json.dumps(options)}")

    free = check_memory()
    print(f"System RAM free: {free} GB")
    if free >= 0 and free < 2.0:
        print("SKIP — low memory")
        return None

    # Unload and reload
    unload_model(MODEL)
    time.sleep(3)

    print("Loading model...")
    start = time.perf_counter()
    warmup = ollama_api("/api/generate", {
        "model": MODEL, "prompt": "Hi", "stream": False, "think": False,
        "options": {**options, "num_predict": 1},
    }, timeout=300)
    load_time = time.perf_counter() - start

    if not warmup:
        print(f"FAILED to load — skipping")
        return None

    if "error" in (warmup or {}):
        print(f"Load error: {warmup['error']}")
        return None

    print(f"Loaded in {load_time:.1f}s")

    mem = get_gpu_memory()
    ps = ollama_api("/api/ps")
    if ps and ps.get("models"):
        m = ps["models"][0]
        vram = m.get("size_vram", 0) / (1024**3)
        total = m.get("size", 0) / (1024**3)
        print(f"VRAM: {vram:.1f} GB / Total: {total:.1f} GB")

    results = []
    for prompt_config in prompts:
        runs = []
        for i in range(2):
            print(f"  [{prompt_config['name']}] run {i+1}/2...", end=" ", flush=True)
            r = run_single_benchmark(MODEL, prompt_config, options)
            if r:
                runs.append(r)
                print(f"{r['tokens_per_sec']} tok/s (prompt: {r['prompt_tokens_per_sec']} tok/s)")
            else:
                print("FAILED")

        if runs:
            avg_gen = sum(r["tokens_per_sec"] for r in runs) / len(runs)
            avg_prompt = sum(r["prompt_tokens_per_sec"] for r in runs) / len(runs)
            results.append({
                "config_name": config_name,
                "prompt_name": prompt_config["name"],
                "avg_tokens_per_sec": round(avg_gen, 1),
                "avg_prompt_tokens_per_sec": round(avg_prompt, 1),
                "runs": runs,
                "memory": mem,
                "options": options,
                "load_time_s": round(load_time, 1),
            })

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Autoresearch Phase 2B: Safe 235B Optimization")
    print(f"GPU: AMD Radeon 8060S (gfx1151, 96 GiB UMA)")
    print(f"Model: {MODEL}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Note: 66/95 layers on GPU, 29 on CPU. ~56 GB VRAM, ~24 GB system RAM.")
    print(f"Avoiding configs that OOM'd in Phase 2B attempt 1 (ctx >= 8192 without flash).")

    # Safe configs — tested or expected to fit
    configs = [
        # Baselines at safe context sizes
        ("baseline_4k", {"num_ctx": 4096}),
        ("baseline_2k", {"num_ctx": 2048}),

        # Batch size (at 4k ctx) — affects prompt throughput
        ("batch_256_4k", {"num_ctx": 4096, "num_batch": 256}),
        ("batch_512_4k", {"num_ctx": 4096, "num_batch": 512}),
        ("batch_1024_4k", {"num_ctx": 4096, "num_batch": 1024}),

        # Flash attention at safe ctx sizes
        ("flash_on_2k", {"num_ctx": 2048, "flash_attention": True}),
        ("flash_on_4k", {"num_ctx": 4096, "flash_attention": True}),

        # Flash + batch combos
        ("flash_batch512_2k", {"num_ctx": 2048, "num_batch": 512, "flash_attention": True}),
        ("flash_batch1024_4k", {"num_ctx": 4096, "num_batch": 1024, "flash_attention": True}),

        # Thread count (affects CPU layers)
        ("threads_8_4k", {"num_ctx": 4096, "num_thread": 8}),
        ("threads_24_4k", {"num_ctx": 4096, "num_thread": 24}),

        # Attempt larger ctx WITH flash (might fit due to reduced KV memory)
        ("flash_8k", {"num_ctx": 8192, "flash_attention": True}),
    ]

    # Use short + medium prompts only (long takes too long per config)
    prompts = BENCH_PROMPTS[:2]

    all_results = []
    for config_name, options in configs:
        try:
            results = run_config(config_name, options, prompts)
            if results:
                all_results.extend(results)
                # Save incrementally
                results_file = RESULTS_DIR / f"235b_safe_opt_{timestamp}.json"
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            print(f"ERROR in {config_name}: {e}")
            # If the runner crashed, wait and try next config
            time.sleep(10)

    # Final save
    results_file = RESULTS_DIR / f"235b_safe_opt_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Print ranked summary
    print(f"\n{'=' * 80}")
    print("235B OPTIMIZATION RESULTS — Ranked by gen tok/s")
    print(f"{'=' * 80}")
    print(f"{'Config':<30} {'Prompt':<15} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Load (s)':>10}")
    print("-" * 80)

    sorted_results = sorted(all_results, key=lambda x: x["avg_tokens_per_sec"], reverse=True)
    for r in sorted_results:
        print(f"{r['config_name']:<30} {r['prompt_name']:<15} {r['avg_tokens_per_sec']:>10.1f} "
              f"{r['avg_prompt_tokens_per_sec']:>12.1f} {r.get('load_time_s', 0):>10.1f}")

    # Overall config ranking
    config_avg = {}
    for r in all_results:
        name = r["config_name"]
        if name not in config_avg:
            config_avg[name] = []
        config_avg[name].append(r["avg_tokens_per_sec"])

    print(f"\n{'=' * 60}")
    print("OVERALL RANKING (avg gen tok/s)")
    print(f"{'=' * 60}")
    ranked = sorted(config_avg.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    for i, (name, scores) in enumerate(ranked, 1):
        avg = sum(scores) / len(scores)
        print(f"  #{i}: {name:<30} {avg:.1f} tok/s")

    if ranked:
        print(f"\n  WINNER: {ranked[0][0]} at {sum(ranked[0][1]) / len(ranked[0][1]):.1f} tok/s")


if __name__ == "__main__":
    main()
