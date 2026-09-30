#!/usr/bin/env bash
set -euo pipefail

backend="${1:?Use param2 or granite}"
limit="${2:-2}"
max_tokens="${3:-4096}"
granite_gpu="${4:-0}"
scenario_timeout="${ITBENCH_TIMEOUT:-600}"
python="$HOME/research/bin/python"
model_dir="$HOME/professional-agent-evals/models/param2-thinking"
dataset_dir="$HOME/professional-agent-evals/ITBench-Lite/snapshots/sre/v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7"
reference_dir="$HOME/professional-agent-evals/itbench-agent-reference"
run_id=$(date +%Y%m%dT%H%M%S)
output="results/itbench-lite-$backend-$run_id/trajectory.jsonl"
mkdir -p "$(dirname "$output")"

if [[ ! -d "$dataset_dir/Scenario-1" ]]; then
  echo "ITBench Lite SRE snapshots are missing." >&2
  exit 1
fi

if [[ "$backend" == "param2" ]]; then
  mapfile -t gpu_status < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i 0,1)
  if (( ${#gpu_status[@]} != 2 )); then
    echo "Expected two GPUs." >&2
    exit 1
  fi
  for status in "${gpu_status[@]}"; do
    IFS=, read -r free_mib utilization <<< "$status"
    if (( free_mib < 22000 || utilization > 50 )); then
      echo "GPU capacity is busy: ${gpu_status[*]}" >&2
      exit 1
    fi
  done
  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$reference_dir:$PWD" \
    "$python" -m eval_smoke.itbench_lite --backend param2 --model "$model_dir" \
    --snapshot-root "$dataset_dir" --limit "$limit" --max-tokens "$max_tokens" \
    --timeout "$scenario_timeout" --output "$output"
elif [[ "$backend" == "granite" ]]; then
  IFS=, read -r free_mib utilization < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "$granite_gpu")
  if (( free_mib < 11000 || utilization > 50 )); then
    echo "GPU $granite_gpu is busy: $free_mib MiB free, $utilization% utilization." >&2
    exit 1
  fi
  port=12177
  if ss -ltn "( sport = :$port )" | grep -q LISTEN; then
    echo "Port $port is occupied." >&2
    exit 1
  fi
  OLLAMA_HOST="127.0.0.1:$port" OLLAMA_MODELS="$HOME/professional-agent-evals/ollama-models" \
    CUDA_VISIBLE_DEVICES="$granite_gpu" OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=16384 \
    "$HOME/professional-agent-evals/ollama-runtime/bin/ollama" serve > "$(dirname "$output")/ollama.log" 2>&1 &
  server_pid=$!
  trap 'kill "$server_pid" 2>/dev/null || true' EXIT
  sleep 3
  PYTHONPATH="$reference_dir:$PWD" "$python" -m eval_smoke.itbench_lite \
    --backend openai --model granite4.2:3b --base-url "http://127.0.0.1:$port/v1" \
    --snapshot-root "$dataset_dir" --limit "$limit" --max-tokens "$max_tokens" \
    --timeout "$scenario_timeout" --output "$output"
else
  echo "Backend must be param2 or granite." >&2
  exit 1
fi
