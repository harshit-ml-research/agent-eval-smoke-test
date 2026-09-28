# Agent evaluation smoke test

Small, backend independent agent evaluation harness for validating a local model before integrating ITBench.

## Run

Use Python 3.11 with `openai`, `httpx`, `pydantic`, and `pytest` installed. Serve an OpenAI compatible chat endpoint, then run:

```bash
python -m eval_smoke.run --base-url http://127.0.0.1:11434/v1 --model granite4.2:3b --output results/smoke.jsonl
```

The Ollama endpoint is one option. Any OpenAI compatible endpoint, including vLLM, can be selected with `--base-url` and `--model`. Requests, responses, trajectories, errors, scores, and aggregate metrics are saved as JSONL or JSON files under `results/`.

## Layout

`eval_smoke/models` handles model requests. `harness` runs the agent and tools. `benchmarks` supplies tasks and scoring. `logging` writes results. `analysis` calculates metrics. The ITBench adapter is intentionally a placeholder.
