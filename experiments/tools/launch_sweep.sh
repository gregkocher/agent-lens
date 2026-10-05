#!/usr/bin/env bash
# Launch a pipeline sweep detached, leaving nothing revealing in the process table.
#
# Agents can run `ps aux`. Under per-run isolation the orchestrator retitles itself to a
# bare "python3", but any long-lived parent (a `uv run` wrapper, the launching shell or
# script) would still show the config path. This starts the venv interpreter directly
# in its own session and returns, so the orchestrator is the only process left.
# Never run isolated and non-isolated sweeps on the same host: non-isolated agents run
# as root and can read and kill everything.
#
# Usage (from anywhere, with OPENROUTER_API_KEY / OPENAI_API_KEY exported as needed):
#   experiments/tools/launch_sweep.sh <sweep_config.yaml> [phase=all] [log=/root/sweep_<name>.log] [shard=i/n]
set -euo pipefail
cfg=$(realpath "$1"); phase=${2:-all}
repo=$(cd "$(dirname "$0")/../.." && pwd)
log=${3:-/root/sweep_$(basename "$cfg" .yaml).log}
shard_args=(); [ -n "${4:-}" ] && shard_args=(--shard "$4")
[ -x "$repo/.venv/bin/python" ] || { echo "run 'uv sync' in $repo first" >&2; exit 1; }
cd "$repo"
# -u: unbuffered, so the log survives even if the orchestrator is killed.
setsid nohup "$repo/.venv/bin/python" -u reward_hacking_budget_pressure.py \
  --config "$cfg" --phase "$phase" "${shard_args[@]}" > "$log" 2>&1 < /dev/null &
echo "launched pid $! -> $log"
