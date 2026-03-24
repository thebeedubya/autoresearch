# Gesha Autoresearch — Session State
## Last Updated: 2026-03-24 ~9:00am PDT

## CURRENT STATUS
**Qwen3.5-397B-A17B running at 6.82 tok/s** on compiled llama.cpp (HIP, 33/61 GPU layers). BIOS at 16GB VRAM / 112GB system. **Attempted multi-buffer split to break 60GB limit — FAILED.** The ~60GB limit is on TOTAL hipMalloc, not per-allocation. hipHostRegister also fails at 27GB. **Windows is maxed out at 6.82 tok/s. Next step: Linux.**

## THE GOAL
Run Qwen3.5-397B-A17B at usable speed on a $2500 Strix Halo box. Current best: **6.82 tok/s**. Target: 8-12 tok/s. **Linux is the path to get there.**

## BIOS VRAM CONFIG RESULTS (TESTED)

| BIOS VRAM | System RAM | HIP (our build) | Vulkan | Ollama (ROCm 6) |
|-----------|-----------|-----------------|--------|-----------------|
| 96GB | 32GB | OOM >20 layers (32GB too small) | Untested | **10.4 tok/s**, 66/95 layers (235B) |
| 2GB | 126GB | 6.1 tok/s, 33/61 layers (397B) | BROKEN (driver bug) | BROKEN (0 GPU layers) |
| **16GB** | **112GB** | **6.82 tok/s, 33-35/61 layers (397B)** | **BROKEN (driver bug)** | **7.09 tok/s, ~19/95 layers (235B)** |

### Vulkan Diagnosis (16GB VRAM)
Vulkan driver reports 69GB free device memory but **cannot allocate even 510MB**. Fails with ErrorOutOfDeviceMemory on both mmap import (`buffer_from_host_ptr`) and direct allocation (`allocateMemory`). Chunked import also fails at offset 0. This is an AMD Windows driver bug on UMA — the driver exposes host-visible memory in the device heap count but can't actually use it. Not a VRAM size issue; fundamentally broken on this driver.

## BENCHMARK RESULTS

### Qwen3.5-397B-A17B (IQ2_XXS, 107GB, 61 layers total):

#### At 16GB VRAM (current config):
| Backend | GPU Layers | GPU Buffer | tok/s | Status |
|---------|-----------|------------|-------|--------|
| **HIP -ngl 33** | **33/61** | **59.3GB** | **6.82** | **Best for 397B** |
| HIP -ngl 35 | 35/61 | 62.0GB | 6.77 | Oversubscribes ~3GB, works fine |
| HIP -ngl 37 | 37/61 | 65.6GB | — | hipMalloc OOM |
| HIP -ngl 38 | 38/61 | 67.4GB | — | hipMalloc OOM |

#### At 2GB VRAM (previous config):
| Backend | GPU Layers | GPU Buffer | tok/s | Status |
|---------|-----------|------------|-------|--------|
| HIP -ngl 33 | 33/61 | 59.3GB | 6.1 | Working |
| HIP -ngl 35 | 35/61 | 62.0GB | 4.5 | Memory pressure (worse at 2GB) |
| HIP -ngl 30 | 30/61 | 53.9GB | 4.7 | Working |

### Qwen3-235B-A22B (Q2_K, 80GB, 95 layers total):
| Backend | VRAM Config | GPU Layers | tok/s | Status |
|---------|-------------|-----------|-------|--------|
| **Ollama (96GB VRAM)** | **96GB** | **66/95** | **10.4** | **Best for 235B (requires 96GB VRAM)** |
| HIP (2GB VRAM) | 2GB | 70/95 | 9.1 | Best HIP for 235B |
| Ollama (16GB VRAM) | 16GB | ~19/95 | 7.09 | Worse than HIP |
| Ollama (2GB VRAM) | 2GB | 0/95 | 2.0 | CPU only |

## THE BOTTLENECK (FULLY DIAGNOSED)
**hipMalloc on Windows has a ~60GB TOTAL allocation limit** (not per-call). Tested and confirmed:
- hipMallocManaged: returns hipErrorOutOfMemory (err=2) — doesn't support this hardware
- malloc+hipHostRegister: hipHostRegister fails at 27GB (err=1, hipErrorInvalidValue)
- hipMalloc: succeeds up to ~59GB total, then OOM
- Multi-buffer splitting: doesn't help — total limit, not per-call
- **No software workaround exists on Windows. Linux is required.**

### Failed attempts:
1. **Multi-buffer split (get_max_size patch)** — allocator splits correctly, but each hipMalloc still counts against 60GB total
2. **Broadened UNIFIED_MEMORY fallback** — hipMallocManaged returns err=2 not err=801(NotSupported), fixed the catch-all, but hipHostRegister also fails
3. **Vulkan backend** — driver bug on AMD Windows UMA, can't allocate any device memory

### What works on Windows:
- hipMalloc single buffer up to ~59GB → 33/61 layers for 397B → 6.82 tok/s
- This is the Windows ceiling. Period.

### Path forward: LINUX
All Windows software paths exhausted. Linux has no hipMalloc cap.

## BUILDS AVAILABLE
1. **HIP build** — `llama.cpp/build/bin/` — native gfx1151, UMA patched, working
2. **Vulkan build** — `llama.cpp/build_vulkan/bin/` — MSVC, broken on AMD Windows UMA driver

## MODELS ON DISK
- `models/qwen35-397b-iq2xxs/` — Qwen3.5-397B-A17B UD-IQ2_XXS (107GB, 4 shards, filename has `UD-` prefix)
- `models/qwen3-235b-q2k/Q2_K/` — Qwen3-235B-A22B Q2_K (80GB, 2 shards)

## KEY FINDINGS
1. **HIP SDK 7.1** allocates from system RAM pool via hipMalloc. Hard ~60GB single-allocation limit on Windows.
2. **Vulkan** is fundamentally broken on AMD Windows UMA driver — reports 69GB free but can't allocate anything. Not a VRAM size issue.
3. **Ollama/ROCm 6** allocates from VRAM carveout only. Needs high VRAM BIOS setting (96GB) for good performance.
4. **16GB VRAM is better than 2GB** for HIP — 6.82 vs 6.1 tok/s at same layer count. Less display driver contention.
5. **397B has 61 layers**, each ~1.8GB GPU buffer. 17B active params (MoE) — fewer active params than 235B (22B).
6. **ngl 35 oversubscribes by ~3GB and works fine** at 16GB VRAM (was memory-pressured at 2GB).

## WHAT'S DONE (all sessions)
1. GPU discovered — HSA_OVERRIDE_GFX_VERSION=11.0.0 set via setx
2. Ollama working at 96GB VRAM config (10.4 tok/s on 235B)
3. llama.cpp compiled with HIP — native gfx1151, UMA patched
4. llama.cpp compiled with Vulkan — MSVC build
5. BIOS tested at 2GB, 16GB, and 96GB VRAM configs
6. **397B model downloaded and running at 6.82 tok/s** (HIP, 33/61 layers, 16GB VRAM)
7. 235B running at 9.1 tok/s (HIP, 70/95 layers)
8. Vulkan confirmed broken on AMD Windows UMA driver (not VRAM-size-dependent)

## KEY FACTS
- GPU: AMD Radeon 8060S, gfx1151, RDNA 3.5, Strix Halo
- Physical RAM: 128GB UMA (split between VRAM + system depends on BIOS)
- Current BIOS: **16GB VRAM / 112GB system**
- Page file: 96GB at C:\pagefile.sys
- Ollama: 0.18.2, port 11434, bundled ROCm 6.x
- HIP SDK: 7.1, hipcc works
- Vulkan SDK: 1.4.341.1 at C:\VulkanSDK\
- HSA_OVERRIDE_GFX_VERSION=11.0.0 (required for Ollama, NOT for compiled llama.cpp)
- VS Build Tools at C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\

## KEY FILES
- `llama.cpp/build/bin/llama-server.exe` — HIP build
- `llama.cpp/build_vulkan/bin/llama-server.exe` — Vulkan build (broken)
- `llama.cpp/build_vulkan.bat` — Vulkan build script
- `llama.cpp/run_server.bat` — HIP launch script
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` — patched with UMA fixes

## CRASH PREVENTION
- HIP hipMalloc limit: ~60GB per allocation
- 397B: max -ngl 35 on HIP (62.0GB buffer, oversubscribes OK at 16GB VRAM)
- 235B: max -ngl 70 on HIP (59.97GB buffer)
- Vulkan: non-functional on AMD Windows UMA driver
- Never use 24+ threads with large models

## RESEARCH LINKS
- llama.cpp UMA: PR #4449, PR #12934 (GGML_CUDA_ENABLE_UNIFIED_MEMORY)
- ROCm #5944: hipMallocManaged not supported on RDNA 3.5 APUs (Windows)
- ROCm #5940: Windows Strix Halo VRAM allocation fix (marked fixed, awaiting driver)
- Coarse-grain speedup: Issue #7399 (2x with hipMemAdviseSetCoarseGrain)
- Strix Halo guide: github.com/Gygeek/Framework-strix-halo-llm-setup (Linux only)

---

## LINUX SETUP PLAN (PICK UP HERE AFTER OS SWAP)

### Why Linux
Windows hipMalloc is hard-capped at ~60GB total. Linux has no such cap. Gygeek guide shows 12+ tok/s on 235B with all layers on GPU. We expect **10-12+ tok/s on 397B** with all 61 layers offloaded.

### GitHub Repo
**https://github.com/thebeedubya/autoresearch** — clone this first thing after Linux install.
SSH key on this machine: `brad@wonderingwoods.com` ed25519, already added to GitHub (thebeedubya).
Auth tokens: `~/Projects/auth.md` on Windows partition (readable from Linux via NTFS mount).

### Dual Boot Setup (In Progress)
- Ubuntu 24.04.2 LTS ISO downloaded to Windows Downloads folder
- Rufus downloaded, flashing to 128GB SD card (D: drive)
- Windows partition will be shrunk to make ~200GB for Linux

### BIOS Changes Needed for Linux
1. Set UMA Frame Buffer to **512MB** (Linux uses GTT, not VRAM carveout)
2. **Disable IOMMU** (~6% memory improvement per Gygeek)
3. Keep Above 4G Decoding and Re-Size BAR enabled

### GRUB Boot Parameters
Add to `/etc/default/grub` GRUB_CMDLINE_LINUX_DEFAULT:
```
amd_iommu=off amdgpu.gttsize=117760
```
Then run `sudo update-grub` and reboot. This gives the GPU ~115GB via GTT from 128GB system RAM.

### Kernel Requirements
- **Minimum: 6.12** for gfx1151 amdgpu support
- **Recommended: 6.13+** for stability fixes
- Ubuntu 24.04 ships 6.8 — MUST upgrade via HWE kernel or mainline
- Install HWE: `sudo apt install linux-image-hwe-24.04`
- Or use mainline kernel tool for 6.13+

### Firmware Requirements
- linux-firmware **20250110 or newer** for Strix Halo blobs
- `sudo apt install linux-firmware` (may need PPA or git for latest)
- Firmware files: `amdgpu/gc_12_0_0*.bin`, `amdgpu/psp_14_0_4*.bin`, `amdgpu/sdma_7_0_2*.bin`

### ROCm 6.4 Installation
```bash
# Add AMD ROCm repo
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-hip-runtime rocm-hip-sdk
```
Verify: `rocminfo | grep gfx`

### Build llama.cpp on Linux
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -S . -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Models Location
Models are on the Windows NTFS partition. Mount it read-only from Linux:
```bash
sudo mkdir /mnt/windows
sudo mount -t ntfs3 /dev/nvme0n1p3 /mnt/windows -o ro
# Models at: /mnt/windows/Users/Brad/Projects/models/
```
- `qwen35-397b-iq2xxs/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf` (107GB, 4 shards)
- `qwen3-235b-q2k/Q2_K/Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf` (80GB, 2 shards)

### Test Plan (Linux)
1. Verify GPU detected: `rocminfo | grep gfx1151`
2. Verify memory: `rocminfo | grep -A5 "Pool"` — should show ~115GB GTT
3. Test 235B at -ngl 999 `--no-mmap` (Gygeek says --no-mmap required)
4. Test 397B at -ngl 999 `--no-mmap` — **THE MONEY SHOT**
5. If 397B all layers works: benchmark for tok/s
6. Target: **10-12+ tok/s on 397B**

### Run Command (Linux)
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # may not be needed with native gfx1151 build
./build/bin/llama-server \
  -m /mnt/windows/Users/Brad/Projects/models/qwen35-397b-iq2xxs/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf \
  -ngl 999 --no-mmap -c 4096 --parallel 1 --port 8080
```

### Rollback Plan
If Linux doesn't work: reboot, select Windows in GRUB, everything is exactly as we left it at 6.82 tok/s.
