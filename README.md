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

## Layout

`eval_smoke/models` handles model requests. `harness` runs the agent and tools. `benchmarks` supplies tasks and scoring. `logging` writes results. `analysis` calculates metrics. The ITBench adapter is intentionally a placeholder.
