# Title

Qwen3.5-397B at 17-19 tok/s on a Strix Halo iGPU — all 61 layers on GPU via Vulkan (not ROCm)

# Post

Running **Qwen3.5-397B-A17B** (IQ2_XXS, 107GB, 4 GGUF shards) at **17-19 tok/s generation** and **25-33 tok/s prompt processing** on a single AMD Ryzen AI Max+ 395 with 128GB unified memory. All 61 layers offloaded to the integrated Radeon 8060S GPU. Total hardware cost: ~$2,500.

**The setup:**

- AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S (gfx1151, RDNA 3.5, 40 CUs)
- 128GB LPDDR5X unified memory
- llama.cpp built with **Vulkan** (Mesa RADV 24.2.8), NOT ROCm/HIP
- Ubuntu, kernel 6.17

**The key finding: use Vulkan, not ROCm.**

I spent a lot of time trying to get this working through ROCm 6.4 / HIP. On Windows, HIP has a hard ~60GB hipMalloc limit that caps you at 33/61 GPU layers (6.82 tok/s). Moved to Linux expecting ROCm to remove that cap. Instead, the HIP runtime straight up segfaults on gfx1151 — null pointer dereference in `libamdhip64.so` regardless of how many layers you try to offload. Even 10 layers crashes. It's a driver bug, not an OOM issue.

On a whim, I rebuilt llama.cpp with `-DGGML_VULKAN=ON -DGGML_HIP=OFF`. Mesa's open-source RADV Vulkan driver handled everything ROCm couldn't. All 61 layers loaded, no crashes, nearly 3x the Windows performance.

**Results comparison:**

| Config | GPU Layers | tok/s |
|--------|-----------|-------|
| Windows, HIP (llama.cpp) | 33/61 | 6.82 |
| Linux, CPU-only | 0/61 | 9.15 |
| **Linux, Vulkan (llama.cpp)** | **61/61** | **17-19** |

**Other things that mattered:**

- Kernel 6.17 deprecated `amdgpu.gttsize`. You need `ttm.pages_limit=30146560` in GRUB to get the full ~115GB GPU memory pool (defaults to ~56GB otherwise).
- The model has to be on ext4 — mmap from NTFS segfaults. Copy it to a native filesystem.
- Always use `-fit off` with llama.cpp on this hardware. The auto-fit mechanism crashes.

If you have a Strix Halo machine and you're fighting ROCm, try Vulkan. The open-source Mesa driver is doing what AMD's own compute stack can't.

Build instructions and full details: [github.com/thebeedubya/autoresearch](https://github.com/thebeedubya/autoresearch)
