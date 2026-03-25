# Strix Halo LLM Gateway — Design Document

## Overview

A local LLM gateway running on a $2,500 Strix Halo mini PC (AMD Ryzen AI Max+ 395, 128GB unified RAM, Radeon 8060S iGPU). Serves multiple models through a single endpoint with intelligent routing, query logging, and a monitoring dashboard.

## Hardware

- AMD Ryzen AI Max+ 395 (Strix Halo), RDNA 3.5 (gfx1151), 40 CUs
- 128GB LPDDR5X unified memory (109GB visible to OS, 16GB BIOS VRAM carveout)
- GPU backend: Vulkan via Mesa RADV 24.2.8 (ROCm/HIP is broken on gfx1151)
- Kernel: 6.17.0-19-generic, TTM pages_limit=30146560 (~115GB GPU pool)

## Architecture

```
Clients (Haze, Claude Code, API consumers)
        |
        v
  Gateway (Python/FastAPI :9090)
  ├── Smart Router (classifies prompts → selects model)
  ├── Reverse Proxy (forwards to llama-server)
  ├── Query Logger (SQLite — full request/response)
  └── Dashboard (web UI — metrics, query log, model status)
        |
        v
  llama-server Router Mode (:8080, localhost only)
  ├── Model Registry (from models.ini)
  ├── LRU Eviction (max 1 model at a time)
  └── Child Processes (spawned per model on ephemeral ports)
        |
        v
  Vulkan / Radeon 8060S iGPU
```

## Model Library

| ID | Model | Size | Active Params | Role | tok/s |
|----|-------|------|--------------|------|-------|
| qwen3-8b-fast | Qwen3-8B Q8_0 | 8.7 GB | 8B (dense) | Extraction, classification, simple Q&A | 28-30 |
| qwen3-coder | Qwen3-Coder-30B-A3B Q8_0 | 32.5 GB | 3.3B (MoE) | Code generation, review, debugging | ~30-50 (est) |
| qwen35-27b-reason | Qwen3.5-27B Claude Distill Q8_0 | 27 GB | 27B (dense) | Deep reasoning, analysis, complex problems | 7-8 |
| qwen35-397b-heavy | Qwen3.5-397B IQ2_XXS | 107 GB | 17B (MoE) | Maximum capability, frontier-class | 17-19 |

Key constraint: `--models-max 1` — only one model loaded at a time due to GPU memory. Swapping takes 30-60 seconds.

## Smart Routing (Current: Rule-Based POC)

Three regex pattern sets score each incoming prompt:

### FAST_PATTERNS (→ qwen3-8b-fast)
`extract`, `classify`, `categorize`, `tag`, `label`, `summarize`, `translate`, `json`, `structured`, `brief`, `short`, `one sentence`, `yes or no`, `format`, `parse`, `convert`

### CODE_PATTERNS (→ qwen3-coder)
Code blocks (```), `def`, `fn`, `func`, `class`, `import`, `write code`, `implement`, `debug`, `refactor`, `fix bug`, `code review`, language names (python, rust, javascript, etc.), `api endpoint`, `http server`, `database`, `algorithm`

### REASON_PATTERNS (→ qwen35-27b-reason)
`explain why`, `analyze`, `compare and`, `evaluate`, `critique`, `step by step`, `reason`, `think through`, `pros cons`, `trade-off`, `implications`, `argue`, `debate`, `philosophy`, `ethics`, `complex`, `nuance`

### Default behavior
- Short prompts (<100 chars) with no pattern matches → fast model
- No matches at all → fast model
- Explicit model name in request → respected, no auto-routing
- `model: "auto"` or empty → classified and routed

## Routing Improvements (Future)

### Tier 1: Enhanced Rules
- Weight system prompt keywords higher than user message
- Detect conversation context (multi-turn coding → stay on coder)
- Detect output format requests (JSON schema → fast model)
- Token budget awareness (short max_tokens → fast model)

### Tier 2: Embedding-Based Classification
- Embed each prompt using a lightweight model
- Compare against pre-computed category centroids
- More robust to paraphrasing than keywords
- Could run on CPU with a tiny embedding model (~100ms overhead)

### Tier 3: LLM-Based Classification
- Use the 8B model itself to classify (one-shot prompt: "Is this a coding task, reasoning task, or simple extraction?")
- Most accurate but adds 1-2 seconds per request
- Only worthwhile if it prevents a wrong model swap (which costs 30-60s)

### Tier 4: Learning Router
- Log all queries with routing decisions + user feedback
- Train a small classifier on accumulated data
- Personalized routing that improves over time

## Query Logging

SQLite database at `/home/brad/gateway.db`:

```sql
CREATE TABLE queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    model TEXT,          -- includes "[auto]" suffix if auto-routed
    endpoint TEXT,       -- /v1/chat/completions, /v1/messages, etc.
    user_message TEXT,   -- last user message (truncated to 500 chars)
    assistant_message TEXT, -- response preview (truncated to 500 chars)
    prompt_tokens INTEGER,
    gen_tokens INTEGER,
    prompt_tps REAL,
    gen_tps REAL,
    wall_time_ms REAL,
    status_code INTEGER
);
```

## Dashboard

Web UI at gateway root (/) showing:
- Active model with status (loaded/loading/unloaded)
- All registered models with roles and descriptions
- Performance cards (avg gen tok/s, avg prompt tok/s, peak, requests, uptime)
- Session totals (tokens generated, tokens processed)
- RAM usage bar
- Generation speed chart over time
- Prompt processing speed chart over time
- Query log table with timestamp, model, user message, response preview, token counts, speeds, wall time
- Click to expand truncated messages

## API Endpoints

### Proxied to llama-server
- `POST /v1/chat/completions` — OpenAI chat API (auto-routed)
- `POST /v1/messages` — Anthropic Messages API (auto-routed)
- `GET /v1/models` — list available models with status

### Gateway-specific
- `GET /api/stats` — dashboard data (models, memory, performance, totals)
- `GET /api/queries?limit=50` — query log from SQLite
- `GET /api/profiles` — model profiles with roles and descriptions
- `POST /api/classify` — test routing classification without sending to model

## Files

```
/home/brad/gateway.py          — gateway service (proxy + logger + dashboard + router)
/home/brad/start-gateway.sh    — startup script (starts router + gateway, verifies health)
/home/brad/models/models.ini   — model presets (per-model context size, parallelism, etc.)
/home/brad/gateway.db          — SQLite query log
/home/brad/models/             — GGUF model files
  qwen3-8b/                    — 8.7 GB
  qwen3-coder-30b/             — 32.5 GB
  qwen35-27b-opus-v2/          — 27 GB
  qwen35-397b-iq2xxs/          — 107 GB (4 shards)
```

## Key Findings

1. **MoE models are faster than smaller dense models** on this hardware — 397B MoE (17B active) runs at 17-19 tok/s while 27B dense runs at 7-8 tok/s. The Coder (30B MoE, 3.3B active) should be the fastest of all.

2. **Thinking models waste tokens on simple tasks** — the 27B Claude distill spent 1987 thinking tokens on "how is the president" and 300+ thinking tokens before outputting `[]` for paper classification. Match model to task.

3. **Vulkan is the only working GPU path** on Strix Halo gfx1151. ROCm 6.4 segfaults, ROCm 7.2 hits SVM memory limits. Mesa RADV handles everything.

4. **Model swap cost is 30-60 seconds** — routing should minimize unnecessary swaps. If the current model can handle the task reasonably, use it rather than swapping to the optimal one.

5. **llama-server has native router mode** — no need to build process management. `--models-preset` + `--models-max 1` handles everything.

## Open Questions

- Should the coder model download be verified (32.5GB, just completed)?
- What's the right threshold for "this task needs a bigger model" vs "the current one is fine"?
- Should we add a warm-up request after model swap to prime the cache?
- How to handle streaming responses through the proxy?
- Should we persist the SQLite across gateway restarts or start fresh?
