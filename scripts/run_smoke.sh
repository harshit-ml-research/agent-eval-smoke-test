#!/usr/bin/env bash
set -euo pipefail

runtime="$HOME/professional-agent-evals/ollama-runtime/bin/ollama"
models="$HOME/professional-agent-evals/ollama-models"
python="$HOME/research/bin/python"
port=12177

if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "Port $port is occupied. No service was started." >&2
  exit 1
fi

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
if (( free_mib < 11000 )); then
  echo "GPU 0 has only $free_mib MiB free. Need at least 11000 MiB." >&2
  exit 1
fi

mkdir -p results
OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$models" CUDA_VISIBLE_DEVICES=0 OLLAMA_MAX_LOADED_MODELS=1 \
  "$runtime" serve > results/ollama.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

sleep 3
run_id=$(date +%Y%m%dT%H%M%S)
"$python" -m eval_smoke.run \
  --base-url "http://127.0.0.1:$port/v1" \
  --model granite4.2:3b \
  --output "results/smoke-$run_id.jsonl"
