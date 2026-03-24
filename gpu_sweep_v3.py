"""
Autoresearch Phase 2B v3: 235B Optimization
Keep model loaded between configs — only unload when context size changes.
Wait longer for memory to free after unloads.
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


def wait_for_memory(target_gb=5.0, timeout=60):
    """Wait for system RAM to free up after model unload."""
    for i in range(timeout // 2):
        free = check_memory()
        if free >= target_gb:
            return free
        time.sleep(2)
    return check_memory()


def load_model(options, timeout=300):
    """Load model and return load time, or None on failure."""
    print(f"  Loading model with {json.dumps(options)}...")
    start = time.perf_counter()
    result = ollama_api("/api/generate", {
        "model": MODEL, "prompt": "Hi", "stream": False, "think": False,
        "options": {**options, "num_predict": 1},
    }, timeout=timeout)
    load_time = time.perf_counter() - start

    if not result or "error" in (result or {}):
        print(f"  FAILED to load: {result}")
        return None, 0

    print(f"  Loaded in {load_time:.1f}s")
    ps = ollama_api("/api/ps")
    if ps and ps.get("models"):
        m = ps["models"][0]
        vram = m.get("size_vram", 0) / (1024**3)
        total = m.get("size", 0) / (1024**3)
        print(f"  VRAM: {vram:.1f} GB / Total: {total:.1f} GB")

    return result, round(load_time, 1)


def benchmark_config_no_reload(config_name, options, prompts):
    """Benchmark without reloading model — just send different prompts.
    The model uses whatever ctx/batch it was loaded with.
    Ollama applies option overrides per-request for some params."""
    print(f"\n--- {config_name} ---")

    mem = get_gpu_memory()
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
            })

    return results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"235b_opt_v3_{timestamp}.json"

    print(f"Autoresearch Phase 2B v3: 235B Optimization")
    print(f"GPU: AMD Radeon 8060S (gfx1151, 96 GiB UMA)")
    print(f"Model: {MODEL}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"Strategy: Load once per context size, sweep batch/threads/flash via request options")

    prompts = BENCH_PROMPTS[:2]  # short_gen + medium_gen
    all_results = []

    # =========================================================
    # GROUP 1: ctx=4096 (load once, vary other params)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("GROUP 1: Context = 4096")
    print(f"{'=' * 60}")

    _, load_time = load_model({"num_ctx": 4096})
    if load_time == 0:
        print("FATAL: Can't load model")
        return

    group1_configs = [
        ("4k_baseline", {"num_ctx": 4096}),
        ("4k_batch256", {"num_ctx": 4096, "num_batch": 256}),
        ("4k_batch512", {"num_ctx": 4096, "num_batch": 512}),
        ("4k_batch1024", {"num_ctx": 4096, "num_batch": 1024}),
        ("4k_threads8", {"num_ctx": 4096, "num_thread": 8}),
        ("4k_threads12", {"num_ctx": 4096, "num_thread": 12}),
        ("4k_threads24", {"num_ctx": 4096, "num_thread": 24}),
    ]

    for config_name, options in group1_configs:
        try:
            results = benchmark_config_no_reload(config_name, options, prompts)
            if results:
                for r in results:
                    r["load_time_s"] = load_time
                all_results.extend(results)
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            print(f"  ERROR: {e}")

    # =========================================================
    # GROUP 2: ctx=4096 with flash attention (reload needed)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("GROUP 2: Context = 4096 + Flash Attention")
    print(f"{'=' * 60}")

    unload_model(MODEL)
    print("Waiting for memory to free...")
    free = wait_for_memory(target_gb=10.0, timeout=30)
    print(f"  RAM free: {free} GB")

    _, load_time = load_model({"num_ctx": 4096, "flash_attention": True})
    if load_time > 0:
        group2_configs = [
            ("4k_flash", {"num_ctx": 4096, "flash_attention": True}),
            ("4k_flash_batch512", {"num_ctx": 4096, "num_batch": 512, "flash_attention": True}),
            ("4k_flash_batch1024", {"num_ctx": 4096, "num_batch": 1024, "flash_attention": True}),
        ]

        for config_name, options in group2_configs:
            try:
                results = benchmark_config_no_reload(config_name, options, prompts)
                if results:
                    for r in results:
                        r["load_time_s"] = load_time
                    all_results.extend(results)
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)
            except Exception as e:
                print(f"  ERROR: {e}")

    # =========================================================
    # GROUP 3: ctx=2048 (reload needed)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("GROUP 3: Context = 2048")
    print(f"{'=' * 60}")

    unload_model(MODEL)
    free = wait_for_memory(target_gb=10.0, timeout=30)
    print(f"  RAM free: {free} GB")

    _, load_time = load_model({"num_ctx": 2048})
    if load_time > 0:
        group3_configs = [
            ("2k_baseline", {"num_ctx": 2048}),
            ("2k_batch512", {"num_ctx": 2048, "num_batch": 512}),
        ]

        for config_name, options in group3_configs:
            try:
                results = benchmark_config_no_reload(config_name, options, prompts)
                if results:
                    for r in results:
                        r["load_time_s"] = load_time
                    all_results.extend(results)
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)
            except Exception as e:
                print(f"  ERROR: {e}")

    # =========================================================
    # GROUP 4: ctx=2048 + flash (reload needed)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("GROUP 4: Context = 2048 + Flash Attention")
    print(f"{'=' * 60}")

    unload_model(MODEL)
    free = wait_for_memory(target_gb=10.0, timeout=30)
    print(f"  RAM free: {free} GB")

    _, load_time = load_model({"num_ctx": 2048, "flash_attention": True})
    if load_time > 0:
        group4_configs = [
            ("2k_flash", {"num_ctx": 2048, "flash_attention": True}),
            ("2k_flash_batch512", {"num_ctx": 2048, "num_batch": 512, "flash_attention": True}),
        ]

        for config_name, options in group4_configs:
            try:
                results = benchmark_config_no_reload(config_name, options, prompts)
                if results:
                    for r in results:
                        r["load_time_s"] = load_time
                    all_results.extend(results)
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2, default=str)
            except Exception as e:
                print(f"  ERROR: {e}")

    # =========================================================
    # GROUP 5: Try ctx=8192 + flash (might fit)
    # =========================================================
    print(f"\n{'=' * 60}")
    print("GROUP 5: Context = 8192 + Flash Attention (experimental)")
    print(f"{'=' * 60}")

    unload_model(MODEL)
    free = wait_for_memory(target_gb=10.0, timeout=30)
    print(f"  RAM free: {free} GB")

    result, load_time = load_model({"num_ctx": 8192, "flash_attention": True}, timeout=300)
    if load_time > 0 and result:
        try:
            results = benchmark_config_no_reload("8k_flash", {"num_ctx": 8192, "flash_attention": True}, prompts)
            if results:
                for r in results:
                    r["load_time_s"] = load_time
                all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Final save
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # =========================================================
    # RESULTS
    # =========================================================
    print(f"\n{'=' * 80}")
    print("235B OPTIMIZATION RESULTS — Ranked by gen tok/s")
    print(f"{'=' * 80}")
    print(f"{'Config':<25} {'Prompt':<15} {'Gen tok/s':>10} {'Prompt tok/s':>12} {'Load (s)':>10}")
    print("-" * 75)

    sorted_results = sorted(all_results, key=lambda x: x["avg_tokens_per_sec"], reverse=True)
    for r in sorted_results:
        print(f"{r['config_name']:<25} {r['prompt_name']:<15} {r['avg_tokens_per_sec']:>10.1f} "
              f"{r['avg_prompt_tokens_per_sec']:>12.1f} {r.get('load_time_s', 0):>10.1f}")

    # Overall config ranking
    config_avg = {}
    for r in all_results:
        name = r["config_name"]
        if name not in config_avg:
            config_avg[name] = {"gen": [], "prompt": []}
        config_avg[name]["gen"].append(r["avg_tokens_per_sec"])
        config_avg[name]["prompt"].append(r["avg_prompt_tokens_per_sec"])

    print(f"\n{'=' * 60}")
    print("OVERALL RANKING (avg gen tok/s)")
    print(f"{'=' * 60}")
    ranked = sorted(config_avg.items(), key=lambda x: sum(x[1]["gen"]) / len(x[1]["gen"]), reverse=True)
    for i, (name, scores) in enumerate(ranked, 1):
        avg_gen = sum(scores["gen"]) / len(scores["gen"])
        avg_prompt = sum(scores["prompt"]) / len(scores["prompt"])
        print(f"  #{i}: {name:<25} gen: {avg_gen:.1f} tok/s  prompt: {avg_prompt:.1f} tok/s")

    if ranked:
        winner = ranked[0][0]
        winner_gen = sum(ranked[0][1]["gen"]) / len(ranked[0][1]["gen"])
        print(f"\n  WINNER: {winner} at {winner_gen:.1f} tok/s")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
