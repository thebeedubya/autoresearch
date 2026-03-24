# PICKUP — Read This First

## SOLVED (2026-03-24): 17-19 tok/s on Vulkan

Qwen3.5-397B-A17B (107GB, IQ2_XXS) running at **17-19 tok/s** with all 61 layers on the Strix Halo iGPU. The key breakthrough: **Vulkan via Mesa RADV, not ROCm/HIP**.

| Setup | tok/s | Notes |
|-------|-------|-------|
| Windows (HIP, 33/61 GPU layers) | 6.82 | TTM limited to 56GB |
| Linux CPU-only (mmap, ext4) | 9.15 | All Zen 5 cores, no GPU |
| **Linux Vulkan (61/61 GPU layers)** | **17-19** | Mesa RADV, full offload |

## Working Command

```bash
cd ~/llama.cpp && ./build-vulkan/bin/llama-server \
  -m "/home/brad/models/qwen35-397b-iq2xxs/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  -ngl 999 -fit off -c 4096 --parallel 1 --port 8080
```

## What We Discovered

1. **TTM fix worked** — `ttm.pages_limit=30146560` in GRUB gave 115GB GPU pool (up from 56GB)
2. **ROCm 6.4 HIP is broken on gfx1151** — segfault in `libamdhip64.so` at any GPU layer count (999, 55, 40, 10, even 0). Null pointer deref at offset 0x18 during tensor loading. Not OOM — a driver bug.
3. **Vulkan bypasses ROCm entirely** — Mesa RADV driver handles gfx1151 perfectly. Built llama.cpp with `-DGGML_VULKAN=ON -DGGML_HIP=OFF`.
4. **Model must be on ext4** — copied 107GB from NTFS to `/home/brad/models/` because mmap segfaults on NTFS. Vulkan build uses mmap (no `--no-mmap` needed).

## Critical Flags
- **Always use `-fit off`** — the auto-fit crashes on this hardware
- **Do NOT use `HSA_OVERRIDE_GFX_VERSION=11.0.0`** — native gfx1151, override crashes
- **Use the Vulkan build** (`build-vulkan/bin/`) not the HIP build (`build/bin/`)

## Setup Steps (After Fresh Boot)

```bash
# 1. Verify TTM
cat /sys/module/ttm/parameters/pages_limit   # expect 30146560

# 2. Enable extra swap (not in fstab)
sudo swapon /swapfile2

# 3. Run the model
cd ~/llama.cpp && ./build-vulkan/bin/llama-server \
  -m "/home/brad/models/qwen35-397b-iq2xxs/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf" \
  -ngl 999 -fit off -c 4096 --parallel 1 --port 8080
```

## Key Facts
- GPU: Radeon 8060S, gfx1151, RDNA 3.5, integrated, 128GB unified memory
- Vulkan: Mesa RADV 24.2.8, Vulkan 1.3.289
- ROCm: 6.4 (broken for GPU inference on gfx1151 — HIP segfaults)
- llama.cpp Vulkan build: ~/llama.cpp/build-vulkan/bin/llama-server
- llama.cpp HIP build (broken): ~/llama.cpp/build/bin/llama-server
- Model (ext4): /home/brad/models/qwen35-397b-iq2xxs/ (4 GGUF shards, 107GB)
- Model (NTFS backup): /mnt/windows/Users/Brad/Projects/models/qwen35-397b-iq2xxs/
- Kernel: 6.17.0-19-generic
- BIOS: 16GB VRAM / 112GB system
- Extra swap: /swapfile2 (32GB) — needs manual swapon
- UEFI boot order: Ubuntu first (fixed via efibootmgr)
- Repo: /home/brad/autoresearch, GitHub: thebeedubya/autoresearch
