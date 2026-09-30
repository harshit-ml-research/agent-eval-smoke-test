from pathlib import Path

from eval_smoke.analysis.metrics import summarize
from eval_smoke.benchmarks.synthetic import SYSTEM_PROMPT, score, tasks
from eval_smoke.logging.jsonl import append_jsonl
from eval_smoke.models.backend import ModelBackend

from .agent import run_agent


def run_suite(backend: ModelBackend, model: str, output: Path, attempts: int = 1, timeout: float = 120, max_tokens: int = 256, suite: str = "smoke") -> dict:
    if suite == "expanded":
        from eval_smoke.benchmarks.expanded import score as task_score, tasks as task_list
    else:
        task_score, task_list = score, tasks
    records = []
    for task in task_list():
        for attempt in range(attempts):
            result = run_agent(backend, model, task.prompt, SYSTEM_PROMPT, task.allow_tools, timeout=timeout, max_tokens=max_tokens)
            record = {"task_id": task.id, "category": task.category, "attempt": attempt + 1, "model": model, "expected": task.expected, "score": task_score(task, result), **result}
            append_jsonl(output, record)
            records.append(record)
    metrics = summarize(records, k=attempts)
    output.with_suffix(".metrics.json").write_text(__import__("json").dumps(metrics, indent=2) + "\n")
    return metrics
