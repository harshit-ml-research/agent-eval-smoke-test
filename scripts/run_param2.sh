#!/usr/bin/env bash
set -euo pipefail

model="$HOME/professional-agent-evals/models/param2-thinking"
python="$HOME/research/bin/python"
suite="${1:-smoke}"
if [[ "$suite" != "smoke" && "$suite" != "expanded" ]]; then
  echo "Suite must be smoke or expanded." >&2
  exit 1
fi
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

if [[ ! -f "$model/model-00004-of-00004.safetensors" ]]; then
  echo "Param2 checkpoint is missing from $model" >&2
  exit 1
fi

mkdir -p results
run_id=$(date +%Y%m%dT%H%M%S)
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false "$python" -m eval_smoke.run \
  --backend param2 \
  --model "$model" \
  --max-tokens 512 \
  --timeout 600 \
  --suite "$suite" \
  --output "results/param2-$suite-$run_id.jsonl"
