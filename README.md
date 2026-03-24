# autoresearch

Running the largest open-source AI models on a $2,500 desktop. No cloud. No datacenter. Just a Strix Halo box with 128GB of unified memory and stubbornness.

## What This Is

A research log and benchmark suite documenting the journey of running **Qwen3.5-397B** (396 billion parameters) and **Qwen3-235B** on an AMD Ryzen AI Max+ 395 (Strix Halo) with 128GB UMA. Everything here is real — the wins, the walls, and the workarounds.

## Current Results

| Model | Parameters | Active Params | Quant | Backend | GPU Layers | tok/s | Platform |
|-------|-----------|--------------|-------|---------|-----------|-------|----------|
| **Qwen3.5-397B-A17B** | 396B | 17B (MoE) | IQ2_XXS | HIP (llama.cpp) | 33/61 | **6.82** | Windows |
| Qwen3-235B-A22B | 235B | 22B (MoE) | Q2_K | Ollama (ROCm 6) | 66/95 | **10.4** | Windows |
| Qwen3-235B-A22B | 235B | 22B (MoE) | Q2_K | HIP (llama.cpp) | 70/95 | **9.1** | Windows |

**6.82 tokens/second on a 397 billion parameter model running locally.** That's a model larger than GPT-3 generating coherent text on hardware you can buy at Micro Center.

## The Hardware

- **CPU/GPU:** AMD Ryzen AI Max+ 395 — Strix Halo, RDNA 3.5 (gfx1151), 40 CUs
- **Memory:** 128GB LPDDR5X unified memory (CPU and GPU share the same pool)
- **Platform:** Framework Desktop (could be any Strix Halo board)
- **Cost:** ~$2,500 total

## The Story So Far

### Phase 1: Discovery (Windows)

Started with Ollama on Windows. Got 10.4 tok/s on 235B with 96GB VRAM BIOS setting. Not bad, but we wanted the 397B model.

### Phase 2: Compiling llama.cpp with HIP

Ollama bundles ROCm 6 which only allocates from the VRAM carveout. We compiled llama.cpp with HIP SDK 7.1 targeting native gfx1151, which can allocate from system RAM via `hipMalloc`. This let us run 397B at 6.82 tok/s.

### Phase 3: Hitting the Wall

The Windows AMD driver has a **hard ~60GB total hipMalloc limit**. We can only fit 33 of 61 layers on GPU. We tried everything to break it:

- **Multi-buffer splitting** — Patched llama.cpp's `get_max_size` to trigger the existing multi-buffer allocator. The split works correctly, but the limit is on total allocated memory, not per-call.
- **hipMallocManaged** — Returns `hipErrorOutOfMemory` (err=2) on this hardware. Not supported.
- **malloc + hipHostRegister** — `hipHostRegister` fails at 27GB with `hipErrorInvalidValue` (err=1).
- **Vulkan backend** — AMD Windows driver reports 69GB free but can't allocate even 510MB. Fundamentally broken on UMA.

**The 60GB ceiling is a Windows driver limitation. No software workaround exists.**

### Phase 4: Linux (Next)

The [Gygeek guide](https://github.com/Gygeek/Framework-strix-halo-llm-setup) shows 12+ tok/s on similar hardware with Linux. Linux doesn't have the hipMalloc cap. With all 61 layers on GPU, we expect **10-12+ tok/s on 397B**.

## BIOS Configuration

The BIOS UMA split is critical. Different backends need different configs:

| BIOS VRAM | System RAM | Best Backend | Best Result |
|-----------|-----------|-------------|------------|
| 96GB | 32GB | Ollama (ROCm 6) | 10.4 tok/s on 235B |
| 16GB | 112GB | HIP (llama.cpp) | 6.82 tok/s on 397B |
| 2GB | 126GB | HIP (llama.cpp) | 6.1 tok/s on 397B |

**For HIP:** Lower VRAM = more system RAM for hipMalloc. 16GB is the sweet spot (less driver contention than 2GB).

**For Ollama/ROCm 6:** Higher VRAM = more GPU layers. 96GB gives 66/95 layers on 235B.

See [BIOS_UMA_GUIDE.md](BIOS_UMA_GUIDE.md) for step-by-step instructions.

## Building llama.cpp for Strix Halo (Windows)

```bash
# Prerequisites: HIP SDK 7.1, CMake, Clang (from HIP SDK)
cd llama.cpp

# HIP build — native gfx1151
cmake -B build -S . -G Ninja \
  -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1151" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j 16
```

### Running inference

```bash
export PATH="/c/Program Files/AMD/ROCm/7.1/bin:$PATH"

# 397B — max 33-35 GPU layers on Windows (60GB hipMalloc limit)
./build/bin/llama-server \
  -m /path/to/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf \
  -ngl 33 -c 2048 --parallel 1 --port 8080

# 235B — max 70 GPU layers on Windows
./build/bin/llama-server \
  -m /path/to/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf \
  -ngl 70 -c 2048 --parallel 1 --port 8080
```

## Key Technical Findings

1. **HIP SDK 7.1 vs ROCm 6:** HIP 7.1 allocates from system RAM via `hipMalloc`. ROCm 6 (Ollama) allocates from VRAM carveout only. Different memory pools, different limits.

2. **The 60GB Wall:** Windows hipMalloc is hard-limited to ~60GB total. This is a driver limitation — `hipMallocManaged` returns OOM, `hipHostRegister` fails, multi-buffer splitting can't help because it's a total limit not per-call.

3. **Vulkan is broken on AMD Windows UMA:** The driver reports 69GB free device memory but can't allocate any of it. Even 510MB fails with `ErrorOutOfDeviceMemory`. Not a VRAM size issue — it's a driver bug.

4. **UMA means same bandwidth:** On Strix Halo, GPU accessing system RAM has the same bandwidth as accessing VRAM — it's all on the same 256-bit LPDDR5X bus. The BIOS split only affects which pool each runtime can allocate from.

5. **MoE is the key:** 397B has 512 experts but only activates 10 per token (17B active). This means frontier-class model quality at inference costs closer to a 17B dense model.

## Repository Contents

- `state.md` — Detailed session state with all benchmark results and technical findings
- `BIOS_UMA_GUIDE.md` — Step-by-step BIOS configuration guide
- `benchmark.py` — Automated benchmark runner
- `gpu_sweep_*.py` — GPU layer sweep scripts for finding optimal -ngl settings
- `model_ladder.py` — Multi-model benchmark ladder
- `results/` — Raw benchmark data (JSON/CSV)

## What's Next

- [ ] Linux dual-boot setup (Ubuntu 24.04 + kernel 6.13+, ROCm 6.4)
- [ ] All 61 layers on GPU for 397B (no hipMalloc limit on Linux)
- [ ] Target: 10-12+ tok/s on 397B
- [ ] Upstream the `get_max_size` and broadened fallback patches to llama.cpp (useful for others hitting this on Windows)

## Related Resources

- [Gygeek's Strix Halo LLM Setup Guide](https://github.com/Gygeek/Framework-strix-halo-llm-setup) (Linux)
- [llama.cpp UMA support](https://github.com/ggml-org/llama.cpp/pull/12934) — `GGML_CUDA_ENABLE_UNIFIED_MEMORY`
- [ROCm #5940](https://github.com/ROCm/ROCm/issues/5940) — Windows Strix Halo VRAM allocation fix
- [ROCm #5944](https://github.com/ROCm/ROCm/issues/5944) — hipMallocManaged not supported on RDNA 3.5 APUs

## License

MIT
