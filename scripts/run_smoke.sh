#!/usr/bin/env bash
set -euo pipefail

runtime="$HOME/professional-agent-evals/ollama-runtime/bin/ollama"
models="$HOME/professional-agent-evals/ollama-models"
python="$HOME/research/bin/python"
port=12177
suite="${1:-smoke}"
if [[ "$suite" != "smoke" && "$suite" != "expanded" ]]; then
  echo "Suite must be smoke or expanded." >&2
  exit 1
fi

if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
  echo "Port $port is occupied. No service was started." >&2
  exit 1
fi

IFS=, read -r free_mib utilization < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0)
if (( free_mib < 11000 || utilization > 50 )); then
  echo "GPU 0 is busy: $free_mib MiB free, $utilization% utilization." >&2
  exit 1
fi

mkdir -p results
OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$models" CUDA_VISIBLE_DEVICES=0 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=4096 \
  "$runtime" serve > results/ollama.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

sleep 3
run_id=$(date +%Y%m%dT%H%M%S)
"$python" -m eval_smoke.run \
  --base-url "http://127.0.0.1:$port/v1" \
  --model granite4.2:3b \
  --suite "$suite" \
  --max-tokens 512 \
  --timeout 600 \
  --output "results/granite-$suite-$run_id.jsonl"
