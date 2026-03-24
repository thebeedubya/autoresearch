# Gesha BIOS UMA Reconfiguration Guide
## Goal: Unlock full 128GB for GPU inference on Windows

### Why This Should Work
On Strix Halo, all 128GB sits on the same memory bus. The BIOS split (VRAM vs system RAM) only controls which pool each compute node can *allocate* from — not the physical bandwidth. The GPU accesses system RAM at the same speed as VRAM because there is no separate VRAM bus. It's true UMA.

Currently: 96GB VRAM / 32GB system. HIP on Windows allocates from the 32GB system side. The 96GB VRAM sits unused because the Windows HIP runtime can't route allocations there.

Target: **16GB VRAM / 112GB system**. HIP allocates from the 112GB system side. The GPU accesses it via UMA at full bandwidth. The 16GB VRAM carveout handles display, KV cache, and compute buffers.

The Level1Techs/Gygeek configs that got 12.3 tok/s used 512MB VRAM on Linux. We're keeping 16GB as a safe margin for Windows display driver overhead.

---

### Step 1: Note Current BIOS Settings
Before changing anything, write down your current values:
- UMA/VRAM allocation: **96GB** (current)
- Anything else you changed from default

### Step 2: Enter BIOS
1. Restart Gesha
2. Press **DEL** or **F2** during POST (depends on your board — try both)
3. Navigate to **Advanced** or **AMD CBS** or **GFX Configuration** section

### Step 3: Change UMA Frame Buffer Size
Look for one of these settings (name varies by BIOS vendor):
- `UMA Frame Buffer Size`
- `VRAM Size`
- `iGPU Memory Size`
- `GFX Configuration > UMA Frame Buffer Size`
- Under `AMD CBS > NBIO Common Options > GFX Configuration`

**Change from 96GB to 16GB**

If 16GB isn't an option, use the closest value:
- Preferred: **16GB**
- Acceptable: **8GB** or **4GB**
- The Gygeek guide uses **512MB** on Linux — but Windows needs more for the display driver

### Step 4: Verify Other Settings
While in BIOS, confirm:
- **Above 4G Decoding**: Enabled (should already be)
- **Re-Size BAR**: Enabled (allows GPU to access large memory regions)
- **IOMMU**: Disabled (Gygeek recommends this for inference performance)
- **Secure Boot**: Doesn't matter for this change

### Step 5: Save and Exit
- Save changes (usually F10)
- System will reboot

### Step 6: Verify After Boot
Open a terminal (Git Bash or PowerShell) and run:

```bash
# Check system RAM (should show ~112GB+ total)
systeminfo | findstr "Total Physical Memory"

# Check GPU VRAM reported
# In PowerShell:
Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM
```

Also check in Task Manager > Performance > GPU — it should show ~16GB dedicated.

### Step 7: Test Ollama
```bash
# Start Ollama and test 235B
ollama run usamakenway/Qwen3-235B-A22B-Q2_K-GGUF:latest "Say hello"
```

Expected: Ollama should now load MORE layers on GPU because hipMalloc can allocate from the 112GB system pool. On UMA, the GPU accesses this at the same speed.

### Step 8: Test Compiled llama.cpp (HIP)
```bash
# Use our compiled build — the one that was OOMing at 56GB
# With 112GB system RAM, hipMalloc(82GB) should succeed
cd C:\Users\Brad\Projects\llama.cpp
run_server.bat
```

Update `run_server.bat` to:
```batch
@echo off
set PATH=C:\Program Files\AMD\ROCm\7.1\bin;%PATH%
set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

"C:\Users\Brad\Projects\llama.cpp\build\bin\llama-server.exe" -m "C:\Users\Brad\Projects\models\qwen3-235b-q2k\Q2_K\Qwen3-235B-A22B-Q2_K-00001-of-00002.gguf" -ngl 999 -c 4096 --parallel 1 --port 8080
```

Expected: ALL 95 layers on GPU. Target: 12+ tok/s.

---

### What Could Go Wrong
1. **Windows display driver needs minimum VRAM** — if 16GB causes display issues, try 32GB
2. **HIP SDK 7.1 still pins pages** — but now there's 112GB of system RAM to pin from, so 82GB model should fit
3. **Ollama's bundled ROCm may behave differently** with the new split — test both Ollama and our compiled build

### Rollback Plan
If anything goes wrong, go back to BIOS and set VRAM back to 96GB. No data loss risk — this only changes memory allocation, not storage.

---

### Expected Results

| Config | System RAM | VRAM | GPU Layers | Expected tok/s |
|--------|-----------|------|------------|----------------|
| Current (96/32) | 32GB | 96GB | 66/95 (Ollama) | 10.4 |
| New (16/112) | 112GB | 16GB | 95/95 | **12+ tok/s** |
| Level1Techs (Linux) | ~127GB | 512MB | 99/99 | 12.3 |
