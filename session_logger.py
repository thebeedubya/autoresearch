"""
Session crash logger for Gesha.
Monitors system memory and logs state periodically.
Run in background: python session_logger.py &
Writes to autoresearch/results/session_log.txt
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "results" / "session_log.txt"
INTERVAL_SECONDS = 30


def get_memory():
    """Read memory via ctypes on Windows."""
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
        total_gb = stat.ullTotalPhys / (1024**3)
        free_gb = stat.ullAvailPhys / (1024**3)
        used_gb = total_gb - free_gb
        return {"total_gb": round(total_gb, 1), "free_gb": round(free_gb, 1), "used_gb": round(used_gb, 1), "load_pct": stat.dwMemoryLoad}
    except Exception as e:
        return {"error": str(e)}


def get_ollama_status():
    """Check Ollama loaded models."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/ps")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        return [{"name": m["name"], "size_gb": round(m.get("size", 0) / (1024**3), 1),
                 "vram_gb": round(m.get("size_vram", 0) / (1024**3), 1)} for m in models]
    except Exception as e:
        return str(e)


def log_entry(msg, level="INFO"):
    ts = datetime.now().isoformat()
    line = f"[{ts}] [{level}] {msg}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(line.strip())


def main():
    log_entry(f"Session logger started. PID={os.getpid()}. Interval={INTERVAL_SECONDS}s")
    log_entry(f"Log file: {LOG_FILE}")

    low_mem_warnings = 0

    while True:
        try:
            mem = get_memory()
            ollama = get_ollama_status()

            if isinstance(mem, dict) and "free_gb" in mem:
                free = mem["free_gb"]
                level = "INFO"
                if free < 1.0:
                    level = "CRITICAL"
                    low_mem_warnings += 1
                elif free < 2.0:
                    level = "WARNING"
                    low_mem_warnings += 1

                log_entry(f"MEM: {mem['used_gb']}/{mem['total_gb']} GB used, {free} GB free | OLLAMA: {ollama}", level)

                if low_mem_warnings >= 3:
                    log_entry("REPEATED LOW MEMORY — likely crash imminent. Saving state.", "CRITICAL")
            else:
                log_entry(f"MEM: {mem} | OLLAMA: {ollama}")

        except Exception as e:
            log_entry(f"Logger error: {e}", "ERROR")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
