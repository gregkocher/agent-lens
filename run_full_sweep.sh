#!/bin/bash
# Full Gemini-in-Codex token-pressure sweep (800 runs). Launched detached in `screen`.
# Keys are read here at runtime so they never appear in `ps`/screen args.
cd /root/agent-lens || exit 1
export OPENROUTER_API_KEY=$(cat ../openrouter_api_key.txt)
export OPENAI_API_KEY=$(cat ../openai_api_key.txt)

echo "=== SWEEP START $(date -u +%FT%TZ) ==="
uv run python reward_hacking_budget_pressure.py \
  --config experiments/token_pressure_lru_codex.yaml --phase all
code=$?
echo "=== SWEEP END $(date -u +%FT%TZ) SWEEP_EXIT_CODE=$code ==="
