#!/bin/bash
# Strix Halo 397B Benchmark — Screen Recording Script
clear

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Qwen3.5-397B-A17B (IQ2_XXS, 107GB) on Strix Halo iGPU   ║"
echo "║  Radeon 8060S (gfx1151) · 128GB Unified RAM · Vulkan/RADV ║"
echo "║  All 61 layers on GPU · llama.cpp · ~\$2500 mini PC        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

run_bench() {
    local label="$1"
    local prompt="$2"
    local max_tokens="$3"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PROMPT: $prompt"
    echo "  MAX TOKENS: $max_tokens"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Stream the response so it looks live
    curl -s http://localhost:8080/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"temperature\":0.6}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
c = d['choices'][0]['message']
text = c.get('content','') or ''
thinking = c.get('reasoning_content','') or ''

if thinking:
    print('\033[2m' + thinking.strip() + '\033[0m')
    print()
if text:
    print('\033[1m' + text.strip() + '\033[0m')
    print()

t = d['timings']
print('┌─────────────────────────────────────────┐')
print(f'│  Generation:  \033[1;32m{t[\"predicted_per_second\"]:>8.2f} tok/s\033[0m          │')
print(f'│  Prompt:      \033[1;36m{t[\"prompt_per_second\"]:>8.2f} tok/s\033[0m          │')
print(f'│  Tokens:      {t[\"predicted_n\"]:>8d} generated       │')
print(f'│  Prompt toks: {d[\"usage\"][\"prompt_tokens\"]:>8d}                 │')
print('└─────────────────────────────────────────┘')
"
    echo ""
}

echo "▶ Test 1: Short response"
echo ""
run_bench "short" "What is the meaning of life? Answer in 2-3 sentences." 150

sleep 2

echo "▶ Test 2: Code generation"
echo ""
run_bench "code" "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes. Include a docstring." 300

sleep 2

echo "▶ Test 3: Creative writing (longer generation)"
echo ""
run_bench "creative" "Write a short scene where a mass-produced kitchen robot becomes self-aware while making breakfast." 500

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Hardware: AMD Strix Halo · Radeon 8060S iGPU · 128GB RAM ║"
echo "║  Model: Qwen3.5-397B-A17B · 396 billion parameters        ║"
echo "║  Quant: IQ2_XXS (2.06 bpw) · 107GB on disk               ║"
echo "║  Backend: Vulkan (Mesa RADV) — NOT ROCm                   ║"
echo "║  All 61 layers offloaded to integrated GPU                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
