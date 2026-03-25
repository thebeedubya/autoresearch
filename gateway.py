#!/usr/bin/env python3
"""Strix Halo LLM Gateway — proxy, logger, dashboard, smart routing"""

import json
import re
import sqlite3
import subprocess
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

LLAMA_URL = "http://127.0.0.1:8080"
DB_PATH = "/home/brad/gateway.db"
START_TIME = time.time()

# --- Smart Router ---

MODEL_PROFILES = {
    "qwen3-8b-fast": {
        "role": "fast",
        "description": "Fast extraction, classification, simple Q&A",
        "strengths": ["json", "extraction", "classification", "tagging", "summarization", "translation"],
    },
    "qwen3-coder": {
        "role": "code",
        "description": "Code generation, review, debugging",
        "strengths": ["code", "programming", "debugging", "refactor", "rust", "python", "javascript", "typescript", "golang"],
    },
    "qwen35-27b-opus": {
        "role": "reason",
        "description": "Deep reasoning, analysis, complex problems",
        "strengths": ["reasoning", "analysis", "math", "logic", "explain", "compare", "evaluate", "critique"],
    },
    "qwen35-397b-heavy": {
        "role": "heavy",
        "description": "Maximum capability, frontier-class (slow to load)",
        "strengths": ["research", "novel", "creative writing", "philosophy"],
    },
}

CODE_PATTERNS = re.compile(
    r'```|def |fn |func |class |import |require\(|#include|function |'
    r'write.*code|implement|debug|refactor|fix.*bug|code review|'
    r'python|rust|javascript|typescript|golang|java|c\+\+|html|css|sql|'
    r'api endpoint|http server|database|algorithm',
    re.IGNORECASE
)

REASON_PATTERNS = re.compile(
    r'explain.*why|analyze|compare.*and|evaluate|critique|'
    r'step.by.step|reason|think.*through|pros.*cons|trade.?off|'
    r'what.*implications|how.*would.*you|argue|debate|'
    r'philosophy|ethics|complex|nuance',
    re.IGNORECASE
)

FAST_PATTERNS = re.compile(
    r'extract|classify|categorize|tag|label|summarize|translate|'
    r'json|structured|list.*of|bullet|brief|short|one.?word|one.?sentence|'
    r'yes.or.no|true.or.false|format|parse|convert|'
    r'paper abstract|academic paper|research paper|findings',
    re.IGNORECASE
)

CLASSIFY_PROMPT = """Classify this request into exactly one category. Reply with ONLY the category name, nothing else.

Categories:
- FAST: extraction, classification, tagging, summarization, translation, simple Q&A, structured output, JSON, short answers
- CODE: code generation, debugging, code review, refactoring, programming in any language, API design, algorithms
- REASON: deep analysis, complex reasoning, multi-step logic, comparisons, evaluations, critiques, philosophy, ethics, creative writing

Request:
{text}

Category:"""


async def classify_with_llm(text):
    """Use the 8B model to classify the request. Returns model ID."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json={
                    "model": "qwen3-8b-fast",
                    "messages": [{"role": "user", "content": CLASSIFY_PROMPT.format(text=text[:500])}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
            )
            result = resp.json()
            answer = result["choices"][0]["message"].get("content", "").strip().upper()
            if "CODE" in answer:
                return "qwen3-coder"
            elif "REASON" in answer:
                return "qwen35-27b-opus"
            else:
                return "qwen3-8b-fast"
    except Exception:
        return None  # fall back to regex


def classify_request_regex(body):
    """Fast regex-based classification as fallback."""
    messages = body.get("messages", [])
    text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text += " " + content
        if msg.get("role") == "system":
            text += " " + content

    code_score = len(CODE_PATTERNS.findall(text))
    reason_score = len(REASON_PATTERNS.findall(text))
    fast_score = len(FAST_PATTERNS.findall(text))

    if len(text.strip()) < 100 and code_score == 0 and reason_score == 0:
        fast_score += 2

    if code_score > reason_score and code_score > fast_score:
        return "qwen3-coder"
    elif reason_score > fast_score:
        return "qwen35-27b-opus"
    elif fast_score > 0:
        return "qwen3-8b-fast"
    return "qwen3-8b-fast"


async def classify_request(body):
    """Classify a request — uses LLM if 8B is loaded, regex fallback otherwise."""
    model = body.get("model", "")

    # If user explicitly picked a known model, respect it
    if model and model in MODEL_PROFILES:
        return model

    # Check if 8B is loaded for LLM classification
    active = get_active_model()
    if active == "qwen3-8b-fast" or is_model_loaded("qwen3-8b-fast"):
        messages = body.get("messages", [])
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        llm_result = await classify_with_llm(text)
        if llm_result:
            return llm_result

    # Fallback to regex
    return classify_request_regex(body)


def is_model_loaded(model_id):
    """Check if a specific model is currently loaded."""
    try:
        r = httpx.get(f"{LLAMA_URL}/v1/models", timeout=2)
        for m in r.json().get("data", []):
            if m["id"] == model_id and m.get("status", {}).get("value") == "loaded":
                return True
    except Exception:
        pass
    return False


def get_active_model():
    """Get currently loaded model from the router."""
    try:
        r = httpx.get(f"{LLAMA_URL}/v1/models", timeout=2)
        for m in r.json().get("data", []):
            if m.get("status", {}).get("value") == "loaded":
                return m["id"]
    except Exception:
        pass
    return None

# --- SQLite ---

def init_db():
    with get_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
            model TEXT,
            endpoint TEXT,
            user_message TEXT,
            assistant_message TEXT,
            prompt_tokens INTEGER DEFAULT 0,
            gen_tokens INTEGER DEFAULT 0,
            prompt_tps REAL DEFAULT 0,
            gen_tps REAL DEFAULT 0,
            wall_time_ms REAL DEFAULT 0,
            status_code INTEGER DEFAULT 0
        )""")

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

init_db()

# --- Helpers ---

def extract_user_message(body):
    try:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "text":
                            return p["text"][:500]
                return str(c)[:500]
    except Exception:
        pass
    return ""

def extract_assistant_message(data):
    try:
        choices = data.get("choices", data.get("content", []))
        if isinstance(choices, list) and choices:
            c = choices[0]
            if isinstance(c, dict):
                msg = c.get("message", c)
                text = msg.get("content", "") or msg.get("text", "")
                if not text:
                    text = (msg.get("reasoning_content", "") or "")[:200] + "..."
                return str(text)[:500]
    except Exception:
        pass
    return ""

def get_gpu():
    def read_sysfs(path):
        try:
            return int(Path(path).read_text().strip())
        except Exception:
            return 0
    return {
        "util": read_sysfs("/sys/class/drm/card1/device/gpu_busy_percent"),
        "temp_c": read_sysfs("/sys/class/drm/card1/device/hwmon/hwmon5/temp1_input") // 1000,
        "power_w": round(read_sysfs("/sys/class/drm/card1/device/hwmon/hwmon5/power1_input") / 1e6, 1),
        "clock_mhz": read_sysfs("/sys/class/drm/card1/device/hwmon/hwmon5/freq1_input") // 1_000_000,
    }


def get_memory():
    try:
        r = subprocess.run(["free", "-b"], capture_output=True, text=True)
        lines = r.stdout.strip().split("\n")
        mem = lines[1].split()
        swap = lines[2].split()
        return {
            "ram_total": int(mem[1]), "ram_used": int(mem[2]),
            "ram_available": int(mem[6]),
            "swap_total": int(swap[1]), "swap_used": int(swap[2]),
        }
    except Exception:
        return None

# --- Proxy ---

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    url = f"{LLAMA_URL}/v1/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body_bytes = await request.body()

    body_json = None
    user_msg = ""
    model = ""
    routed = False
    if body_bytes:
        try:
            body_json = json.loads(body_bytes)
            user_msg = extract_user_message(body_json)
            model = body_json.get("model", "")

            # Smart routing for chat/messages endpoints
            if path in ("chat/completions", "messages") and request.method == "POST":
                target = await classify_request(body_json)
                if target and target != model:
                    body_json["model"] = target
                    model = target
                    routed = True
                    body_bytes = json.dumps(body_json).encode()
        except Exception:
            pass

    start = time.time()

    # Check if we need a model swap
    active = get_active_model()
    swap_happening = model and active and model != active

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.request(
            method=request.method, url=url,
            headers=headers, content=body_bytes,
        )
    wall_ms = (time.time() - start) * 1000

    # Parse response
    p_tok = g_tok = 0
    g_tps = p_tps = 0.0
    assistant_msg = ""
    try:
        rdata = resp.json()
        usage = rdata.get("usage", {})
        p_tok = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        g_tok = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        timings = rdata.get("timings", {})
        g_tps = round(timings.get("predicted_per_second", 0), 1)
        p_tps = round(timings.get("prompt_per_second", 0), 1)
        if not model:
            model = rdata.get("model", "")
        assistant_msg = extract_assistant_message(rdata)
    except Exception:
        rdata = resp.text

    # Log to SQLite (only for POST requests that generate tokens)
    if request.method == "POST" and (p_tok > 0 or g_tok > 0):
        try:
            with get_db() as db:
                db.execute(
                    """INSERT INTO queries (model, endpoint, user_message, assistant_message,
                       prompt_tokens, gen_tokens, prompt_tps, gen_tps, wall_time_ms, status_code)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (model + (" [auto]" if routed else ""), f"/v1/{path}", user_msg, assistant_msg,
                     p_tok, g_tok, p_tps, g_tps, round(wall_ms), resp.status_code)
                )
        except Exception:
            pass

    return JSONResponse(
        content=rdata if isinstance(rdata, dict) else {"raw": rdata},
        status_code=resp.status_code
    )

# --- Dashboard API (must be before catch-all) ---

@app.get("/api/stats")
def api_stats():
    mem = get_memory()
    gpu = get_gpu()

    # Model info from router
    models = []
    active_model = ""
    try:
        r = httpx.get(f"{LLAMA_URL}/v1/models", timeout=2)
        for m in r.json().get("data", []):
            status = m.get("status", {}).get("value", "unknown")
            models.append({"id": m["id"], "status": status})
            if status == "loaded":
                active_model = m["id"]
    except Exception:
        pass

    # Query stats from SQLite
    with get_db() as db:
        row = db.execute(
            "SELECT COUNT(*) as cnt, SUM(prompt_tokens) as pt, SUM(gen_tokens) as gt, "
            "AVG(gen_tps) as avg_gen, AVG(prompt_tps) as avg_prompt, MAX(gen_tps) as peak_gen "
            "FROM queries WHERE gen_tps > 0"
        ).fetchone()
        recent = db.execute(
            "SELECT gen_tps FROM queries WHERE gen_tps > 0 ORDER BY id DESC LIMIT 100"
        ).fetchall()
        recent_prompt = db.execute(
            "SELECT prompt_tps FROM queries WHERE prompt_tps > 0 ORDER BY id DESC LIMIT 100"
        ).fetchall()

    return {
        "uptime": int(time.time() - START_TIME),
        "models": models,
        "active_model": active_model,
        "memory": mem,
        "gpu": gpu,
        "total_requests": row["cnt"] or 0,
        "total_prompt_tokens": row["pt"] or 0,
        "total_gen_tokens": row["gt"] or 0,
        "avg_gen_tps": round(row["avg_gen"] or 0, 1),
        "avg_prompt_tps": round(row["avg_prompt"] or 0, 1),
        "peak_gen_tps": round(row["peak_gen"] or 0, 1),
        "gen_history": [r["gen_tps"] for r in reversed(recent)],
        "prompt_history": [r["prompt_tps"] for r in reversed(recent_prompt)],
    }

@app.get("/api/profiles")
def api_profiles():
    return MODEL_PROFILES

@app.post("/api/classify")
async def api_classify(request: Request):
    body = await request.json()
    target = await classify_request(body)
    active = get_active_model()
    return {
        "target": target,
        "role": MODEL_PROFILES.get(target, {}).get("role", "unknown"),
        "reason": MODEL_PROFILES.get(target, {}).get("description", ""),
        "active": active,
        "swap_needed": target != active if active else True,
    }

@app.get("/api/queries")
def api_queries(limit: int = 50, offset: int = 0):
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM queries ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
    return [dict(r) for r in rows]

# --- Dashboard UI ---

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Strix Halo Gateway</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,system-ui,monospace;padding:20px;max-width:1400px;margin:0 auto}
h1{color:#58a6ff;margin-bottom:20px;font-size:1.4em}
.grid{display:grid;gap:12px;margin-bottom:16px}
.g4{grid-template-columns:repeat(4,1fr)}
.g6{grid-template-columns:repeat(6,1fr)}
.g3{grid-template-columns:repeat(3,1fr)}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card-label{font-size:.7em;color:#8b949e;text-transform:uppercase;letter-spacing:1px}
.card-value{font-size:1.6em;font-weight:bold;margin-top:4px}
.card-sub{font-size:.75em;color:#8b949e;margin-top:2px}
.green{color:#3fb950}.blue{color:#58a6ff}.yellow{color:#d29922}.purple{color:#bc8cff}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.chart-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.chart-card h3{color:#8b949e;font-size:.8em;margin-bottom:10px}
.mem-bar{height:6px;background:#21262d;border-radius:3px;overflow:hidden;margin-top:6px}
.mem-fill{height:100%;border-radius:3px;transition:width .5s}
table{width:100%;border-collapse:collapse}
th{text-align:left;color:#8b949e;font-size:.7em;text-transform:uppercase;padding:6px 8px;border-bottom:1px solid #30363d}
td{padding:6px 8px;border-bottom:1px solid #21262d;font-size:.8em;vertical-align:top}
.trunc{max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;cursor:pointer}
.trunc.expanded{white-space:normal;word-break:break-word}
.q-user{color:#58a6ff}.q-asst{color:#8b949e;font-style:italic}.q-time{color:#484f58;font-size:.75em}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dot-loaded{background:#3fb950}.dot-loading{background:#d29922;animation:pulse 1s infinite}
.dot-unloaded{background:#484f58}
.model-btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:.8em;margin:4px}
.model-btn:hover{background:#30363d}.model-btn.active{border-color:#58a6ff;color:#58a6ff}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
</head><body>

<h1>Strix Halo Gateway</h1>

<div class="card" style="margin-bottom:16px">
  <div class="card-label">Models <span style="font-size:.9em;color:#484f58">(requests auto-route by task type, or set model explicitly)</span></div>
  <div id="model-list" style="margin-top:8px"></div>
</div>

<div class="grid g6">
  <div class="card"><div class="card-label">Status</div><div class="card-value" id="status" style="font-size:1.2em">...</div></div>
  <div class="card"><div class="card-label">Avg Gen</div><div class="card-value blue" id="avg-gen">--</div></div>
  <div class="card"><div class="card-label">Avg Prompt</div><div class="card-value green" id="avg-prompt">--</div></div>
  <div class="card"><div class="card-label">Peak Gen</div><div class="card-value yellow" id="peak-gen">--</div></div>
  <div class="card"><div class="card-label">Requests</div><div class="card-value purple" id="total-req">0</div></div>
  <div class="card"><div class="card-label">Uptime</div><div class="card-value" id="uptime" style="color:#8b949e">--</div></div>
</div>

<div class="grid g3">
  <div class="card"><div class="card-label">Tokens Generated</div><div class="card-value blue" id="total-gen">0</div></div>
  <div class="card"><div class="card-label">Tokens Processed</div><div class="card-value green" id="total-prompt">0</div></div>
  <div class="card">
    <div class="card-label">RAM</div>
    <div class="card-value blue" id="ram-text" style="font-size:1.2em">--</div>
    <div class="mem-bar"><div class="mem-fill" id="ram-bar" style="width:0%;background:#58a6ff"></div></div>
  </div>
</div>

<div class="grid g4">
  <div class="card">
    <div class="card-label">GPU Utilization</div>
    <div class="card-value" id="gpu-util" style="font-size:1.6em">--</div>
    <div class="mem-bar"><div class="mem-fill" id="gpu-util-bar" style="width:0%;background:#3fb950"></div></div>
  </div>
  <div class="card">
    <div class="card-label">GPU Temp</div>
    <div class="card-value" id="gpu-temp" style="font-size:1.6em">--</div>
  </div>
  <div class="card">
    <div class="card-label">GPU Power</div>
    <div class="card-value" id="gpu-power" style="font-size:1.6em">--</div>
  </div>
  <div class="card">
    <div class="card-label">GPU Clock</div>
    <div class="card-value" id="gpu-clock" style="font-size:1.6em">--</div>
  </div>
</div>

<div class="chart-row">
  <div class="chart-card"><h3>Generation Speed (tok/s)</h3><canvas id="genChart" height="180"></canvas></div>
  <div class="chart-card"><h3>Prompt Speed (tok/s)</h3><canvas id="promptChart" height="180"></canvas></div>
</div>

<div class="card">
  <h3 style="color:#8b949e;font-size:.8em;margin-bottom:10px">Query Log</h3>
  <table>
    <thead><tr><th>Time</th><th>Model</th><th>User</th><th>Response</th><th>In</th><th>Out</th><th>Gen</th><th>Prompt</th><th>Wall</th></tr></thead>
    <tbody id="qtable"></tbody>
  </table>
</div>

<script>
const CO={responsive:true,animation:{duration:300},scales:{x:{display:false},y:{grid:{color:'#21262d'},ticks:{color:'#8b949e'},beginAtZero:true}},plugins:{legend:{display:false}}};
const genC=new Chart(document.getElementById('genChart'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#58a6ff',borderWidth:2,fill:true,backgroundColor:'rgba(88,166,255,0.1)',pointRadius:2,tension:.3}]},options:CO});
const proC=new Chart(document.getElementById('promptChart'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#3fb950',borderWidth:2,fill:true,backgroundColor:'rgba(63,185,80,0.1)',pointRadius:2,tension:.3}]},options:CO});

function fmt(b){return(b/1024**3).toFixed(1)+' GB'}
function fmtUp(s){const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h>0?h+'h '+m+'m':m+'m'}
function esc(s){const d=document.createElement('div');d.textContent=s||'';return d.innerHTML}

document.addEventListener('click',e=>{if(e.target.classList.contains('trunc'))e.target.classList.toggle('expanded')});

async function refresh(){
  try{
    const[stats,queries]=await Promise.all([fetch('/api/stats').then(r=>r.json()),fetch('/api/queries?limit=30').then(r=>r.json())]);

    // Models
    const ml=document.getElementById('model-list');
    const profiles=await fetch('/api/profiles').then(r=>r.json()).catch(()=>({}));
    ml.innerHTML=stats.models.map(m=>{
      const cls=m.status==='loaded'?'dot-loaded':m.status==='loading'?'dot-loading':'dot-unloaded';
      const p=profiles[m.id]||{};
      const role=p.role?`<span style="color:#484f58;font-size:.8em">[${p.role}]</span>`:'';
      const desc=p.description?`<span style="color:#484f58;font-size:.75em"> — ${p.description}</span>`:'';
      return `<div style="margin:4px 0"><span class="dot ${cls}"></span><strong>${m.id}</strong> ${role}${desc} <span style="color:#484f58;font-size:.75em">(${m.status})</span></div>`;
    }).join('');

    // Status
    const active=stats.active_model;
    document.getElementById('status').innerHTML=active?`<span class="dot dot-loaded"></span>${active}`:'<span class="dot dot-unloaded"></span>No model loaded';

    // Stats
    document.getElementById('avg-gen').textContent=stats.avg_gen_tps+' tok/s';
    document.getElementById('avg-prompt').textContent=stats.avg_prompt_tps+' tok/s';
    document.getElementById('peak-gen').textContent=stats.peak_gen_tps+' tok/s';
    document.getElementById('total-req').textContent=stats.total_requests;
    document.getElementById('uptime').textContent=fmtUp(stats.uptime);
    document.getElementById('total-gen').textContent=(stats.total_gen_tokens||0).toLocaleString();
    document.getElementById('total-prompt').textContent=(stats.total_prompt_tokens||0).toLocaleString();

    if(stats.memory){
      const m=stats.memory,pct=((m.ram_total-m.ram_available)/m.ram_total*100).toFixed(0);
      document.getElementById('ram-text').textContent=fmt(m.ram_total-m.ram_available)+' / '+fmt(m.ram_total);
      document.getElementById('ram-bar').style.width=pct+'%';
    }

    if(stats.gpu){
      const g=stats.gpu;
      const utilEl=document.getElementById('gpu-util');
      utilEl.textContent=g.util+'%';
      utilEl.style.color=g.util>80?'#3fb950':g.util>30?'#d29922':'#8b949e';
      document.getElementById('gpu-util-bar').style.width=g.util+'%';
      document.getElementById('gpu-util-bar').style.background=g.util>80?'#3fb950':g.util>30?'#d29922':'#484f58';
      const tempEl=document.getElementById('gpu-temp');
      tempEl.textContent=g.temp_c+'\u00b0C';
      tempEl.style.color=g.temp_c>85?'#f85149':g.temp_c>70?'#d29922':'#3fb950';
      document.getElementById('gpu-power').textContent=g.power_w+'W';
      document.getElementById('gpu-power').style.color='#d29922';
      document.getElementById('gpu-clock').textContent=g.clock_mhz+' MHz';
      document.getElementById('gpu-clock').style.color='#8b949e';
    }

    // Charts
    genC.data.labels=stats.gen_history.map((_,i)=>i);
    genC.data.datasets[0].data=stats.gen_history;genC.update();
    proC.data.labels=stats.prompt_history.map((_,i)=>i);
    proC.data.datasets[0].data=stats.prompt_history;proC.update();

    // Query log
    document.getElementById('qtable').innerHTML=queries.map(q=>`<tr>
      <td class="q-time">${(q.timestamp||'').split('T')[1]||''}</td>
      <td style="font-size:.75em">${esc(q.model)}</td>
      <td class="trunc q-user" title="${esc(q.user_message)}">${esc(q.user_message)}</td>
      <td class="trunc q-asst" title="${esc(q.assistant_message)}">${esc(q.assistant_message)}</td>
      <td>${q.prompt_tokens}</td><td>${q.gen_tokens}</td>
      <td>${q.gen_tps} t/s</td><td>${q.prompt_tps} t/s</td>
      <td>${(q.wall_time_ms/1000).toFixed(1)}s</td>
    </tr>`).join('');
  }catch(e){console.error(e)}
}
refresh();setInterval(refresh,2000);
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
