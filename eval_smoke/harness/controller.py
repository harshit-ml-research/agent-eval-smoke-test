from pathlib import Path

from eval_smoke.analysis.metrics import summarize
from eval_smoke.benchmarks.synthetic import SYSTEM_PROMPT, score, tasks
from eval_smoke.logging.jsonl import append_jsonl
from eval_smoke.models.backend import ModelBackend

from .agent import run_agent


def run_suite(backend: ModelBackend, model: str, output: Path, attempts: int = 1) -> dict:
    records = []
    for task in tasks():
        for attempt in range(attempts):
            result = run_agent(backend, model, task.prompt, SYSTEM_PROMPT, task.allow_tools)
            record = {"task_id": task.id, "attempt": attempt + 1, "model": model, "expected": task.expected, "score": score(task, result), **result}
            append_jsonl(output, record)
            records.append(record)
    metrics = summarize(records, k=attempts)
    output.with_suffix(".metrics.json").write_text(__import__("json").dumps(metrics, indent=2) + "\n")
    return metrics
