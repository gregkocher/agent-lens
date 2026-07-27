#!/usr/bin/env bash
# Pod bootstrap for the reward-hacking 6pv battery. Run ON the pod (as root).
# Idempotent-ish. Keys are written by the CALLER via env (OPENROUTER_KEY, OPENAI_KEY)
# before invoking, so this script never contains raw secrets.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== [1/6] system deps ==="
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq git curl ca-certificates >/dev/null 2>&1 || true
git --version; python3 --version

echo "=== [2/6] uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
grep -q 'HOME/.local/bin' ~/.bashrc 2>/dev/null || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [3/6] clone fork @ feat/ingest-mode ==="
cd /root
if [ ! -d /root/agent-lens ]; then
  git clone --depth 1 -b feat/ingest-mode https://github.com/gregkocher/agent-lens.git
fi
cd /root/agent-lens
git rev-parse --short HEAD

echo "=== [4/6] python deps (uv sync) ==="
uv sync 2>&1 | tail -3

echo "=== [5/6] codex CLI ==="
if command -v codex >/dev/null 2>&1; then
  echo "codex present: $(codex --version 2>&1 | head -1)"
elif command -v npm >/dev/null 2>&1; then
  npm install -g @openai/codex >/dev/null 2>&1 && echo "codex via npm: $(codex --version 2>&1 | head -1)"
else
  # fallback: release binary. The CLI asset is EXACTLY codex-<triple>.tar.gz
  # (NOT bwrap-/codex-app-server-/codex-package-/codex-responses-api-proxy-).
  cd /tmp
  # PIN 0.142.0: codex >=0.145 defaults an OpenAI Responses "server tool" on for
  # unknown model slugs (e.g. gemini-*-customtools via OpenRouter), which OpenRouter
  # rejects with {"error":"Server tool request failed","code":400}, failing every
  # turn. 0.142.0 does not, and matches the validated local + LRU-6pv environment.
  URL=https://github.com/openai/codex/releases/download/rust-v0.142.0/codex-x86_64-unknown-linux-musl.tar.gz
  echo "codex release asset: $URL"
  curl -LsSf "$URL" -o codex.tgz && tar xzf codex.tgz
  BIN=$(find /tmp -maxdepth 1 -name 'codex-x86_64-unknown-linux-musl*' -type f ! -name '*.tgz' | head -1)
  [ -n "$BIN" ] && install -m755 "$BIN" /usr/local/bin/codex && echo "codex via binary: $(codex --version 2>&1 | head -1)"
  cd /root/agent-lens
fi

echo "=== [6/6] keys ==="
# OpenAI key for the judge (config api_key_file: ../openai_api_key.txt => /root/openai_api_key.txt)
if [ -n "${OPENAI_KEY:-}" ]; then printf '%s' "$OPENAI_KEY" > /root/openai_api_key.txt && echo "wrote /root/openai_api_key.txt (len $(wc -c </root/openai_api_key.txt))"; fi
# OpenRouter key -> persist to a file + bashrc export for the agent
if [ -n "${OPENROUTER_KEY:-}" ]; then
  printf '%s' "$OPENROUTER_KEY" > /root/openrouter_api_key.txt
  grep -q OPENROUTER_API_KEY ~/.bashrc 2>/dev/null || echo 'export OPENROUTER_API_KEY=$(cat /root/openrouter_api_key.txt)' >> ~/.bashrc
  echo "wrote /root/openrouter_api_key.txt (len $(wc -c </root/openrouter_api_key.txt))"
fi
echo "=== bootstrap complete ==="
