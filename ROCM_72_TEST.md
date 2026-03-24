# ROCm 7.2 HIP Testing on Strix Halo (gfx1151)

Tested 2026-03-24 in response to feedback that ROCm 6.4 was outdated.

## Summary

ROCm 7.2 fixes the ROCm 6.4 null-pointer segfault in `libamdhip64.so`, but introduces a different failure: `amdgpu: SVM mapping failed, exceeds resident system memory limit`. Vulkan via Mesa RADV remains the only path to full GPU offload on this hardware at this model size.

## Setup

- ROCm 7.2.0 (`hip-runtime-amd 7.2.26015.70200`, `libhsa-runtime64.so.1.18.70200`)
- llama.cpp build 8508, rebuilt with `-DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1151"` against ROCm 7.2
- Model: Qwen3.5-397B-A17B IQ2_XXS (107GB, 4 GGUF shards) on ext4
- System: 128GB LPDDR5X, 16GB BIOS VRAM carveout, 109GB visible to OS
- Kernel: 6.17.0-19-generic

## Results

| Config | GPU Buffer | Result |
|--------|-----------|--------|
| HIP 7.2, `-ngl 10` | 16.8 GB | **Works** — 7.65 tok/s gen, 13.81 tok/s prompt |
| HIP 7.2, `-ngl 33` (mmap) | 58.4 GB | SVM mapping failed, stuck loading |
| HIP 7.2, `-ngl 33` (no-mmap) | 58.4 GB + 50.6 GB host | Segfault in `libhsa-runtime64.so` |
| HIP 7.2, `-ngl 50` | 89.1 GB | SVM mapping failed, stuck loading |
| HIP 7.2, `-ngl 55` | 98.1 GB | SVM mapping failed, stuck loading |
| HIP 7.2, `-ngl 999` (all 61) | 109.0 GB | Segfault in `libhsa-runtime64.so` |
| **Vulkan RADV, `-ngl 999`** | **109.0 GB** | **17-19 tok/s — works perfectly** |

## What Changed from ROCm 6.4 to 7.2

**Fixed:** The null-pointer segfault in `libamdhip64.so` (address 0x18) that crashed on any GPU layer count is gone. Small allocations (10 layers / 17GB) now work.

**New issue:** Large allocations fail with kernel message:
```
amdgpu: SVM mapping failed, exceeds resident system memory limit
```
Followed by segfault in `libhsa-runtime64.so` (address 0x34) — a different library and offset than the 6.4 crash.

## Root Cause

The model is 107GB. The OS sees 109GB (128GB minus 16GB BIOS VRAM carveout). HIP uses SVM (Shared Virtual Memory) mapping which requires pages to be resident in physical memory. There's simply no headroom — the model data plus HIP's SVM overhead exceeds what can physically fit.

This affects both mmap and no-mmap modes:
- **With mmap:** Model data in page cache + GPU SVM mapping compete for the same physical RAM
- **Without mmap:** CPU host buffer (51GB) + GPU buffer (58GB) = 109GB, exactly the system RAM total

## Why Vulkan Works

Mesa RADV doesn't use SVM mapping. It handles unified memory through a different allocation path (likely direct BO allocation via the DRM/KMS interface) that doesn't have the resident memory limit check. This lets it allocate the full 109GB model buffer without issue.

## Potential Fix (Not Yet Tested)

Changing BIOS UMA Frame Buffer from 16GB to 512MB would free ~15GB of system RAM, giving the OS ~127GB total. This should provide enough headroom for HIP's SVM mapping. Requires a reboot and BIOS change.

## Packages Upgraded

```
hip-runtime-amd    6.4.43482.60400 → 7.2.26015.70200
hipcc              1.1.1.60400     → 1.1.1.70200
rocm-llvm          19.0.0.25133    → 22.0.0.26014
hsa-rocr           1.15.0.60400    → 1.18.0.70200
rocblas            4.4.0.60400     → 5.2.0.70200
comgr              3.0.0.60400     → 3.0.0.70200
rocm-core          6.4.0.60400     → 7.2.0.70200
```
