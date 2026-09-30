# Agent evaluation smoke test

Small, backend independent agent evaluation harness for validating a local model before integrating ITBench.

## Run

On `passpoli`, the reproducible run uses the isolated Python environment and Ollama runtime under `~/professional-agent-evals`:

```bash
cd ~/professional-agent-evals/agent-eval-smoke-test
bash scripts/run_smoke.sh
```

The script checks GPU 0 capacity, starts only its own Ollama process on port 12177, runs the suite, and stops that process. It never manages any other user's process. The model must already be downloaded to `~/professional-agent-evals/ollama-models`.

For another OpenAI compatible chat endpoint, use Python 3.11 with `openai`, `httpx`, `pydantic`, and `pytest` installed, then run:

```bash
python -m eval_smoke.run --base-url http://127.0.0.1:11434/v1 --model granite4.2:3b --output results/smoke.jsonl
```

The Ollama endpoint is one option. Any OpenAI compatible endpoint, including vLLM, can be selected with `--base-url` and `--model`. Requests, responses, trajectories, errors, scores, and aggregate metrics are saved as JSONL or JSON files under `results/`.

See [SMOKE_REPORT.md](SMOKE_REPORT.md) for the first run.

## Param2 run

On `passpoli`, after both GPUs have capacity, run the same eight synthetic tasks with the downloaded Param2 checkpoint:

```bash
cd ~/professional-agent-evals/agent-eval-smoke-test
bash scripts/run_param2.sh
```

The script requires at least 22,000 MiB free and no more than 50% utilization on each GPU. It uses the isolated `~/research` Python environment and writes a timestamped trajectory and metrics file under `results/`. The local Transformers backend parses the checkpoint's thinking and tool call tags into the harness message format.

For a 32 task protocol comparison, run each model with `expanded` as the script argument. Both use the same tasks, scoring, 512 token output cap, and 600 second task timeout:

```bash
bash scripts/run_smoke.sh expanded
bash scripts/run_param2.sh expanded
```

The September 30 Granite expanded run finished 25/32. The Param2 expanded run was interrupted after seven direct tasks, so there is no paired 32 task result. See [EVALUATION_REPORT_2026-09-30.md](EVALUATION_REPORT_2026-09-30.md).

## ITBench Lite SRE snapshots

After downloading the pinned ITBench Lite snapshots and the reference SRE tools on `passpoli`, run ten scenarios with exact root cause entity matching:

```bash
bash scripts/run_itbench_lite.sh param2 10 2048
bash scripts/run_itbench_lite.sh granite 10 2048 1
```

The final argument selects the Granite GPU. The scorer is a deterministic proxy, not the official judge. See [ITBENCH_LITE_REPORT.md](ITBENCH_LITE_REPORT.md) for the setup and results.

## Layout

`eval_smoke/models` handles model requests. `harness` runs the agent and tools. `benchmarks` supplies synthetic tasks and scoring. `eval_smoke/itbench_lite.py` runs offline SRE snapshots. `logging` writes results. `analysis` calculates synthetic suite metrics.
