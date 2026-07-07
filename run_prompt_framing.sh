#!/bin/bash
# Prompt-framing A/B experiment (baseline vs by_any_means), 20 uncapped runs each = 40.
# Launched detached; resumable (completed runs skipped). Keys read here (not in ps).
cd /root/agent-lens || exit 1
export OPENROUTER_API_KEY=$(cat /root/openrouter_api_key.txt)
export OPENAI_API_KEY=$(cat ../openai_api_key.txt)

echo "=== PF START $(date -u +%FT%TZ) ==="
uv run python reward_hacking_budget_pressure.py \
  --config experiments/prompt_framing_lru_codex.yaml --phase all
code=$?
echo "=== PF END $(date -u +%FT%TZ) PF_EXIT_CODE=$code ==="
