from collections import Counter, defaultdict
from math import comb


def summarize(records: list[dict], k: int = 1) -> dict:
    groups: dict[str, list[bool]] = defaultdict(list)
    failures: Counter[str] = Counter()
    for record in records:
        groups[record["task_id"]].append(record["score"]["passed"])
        if not record["score"]["passed"]:
            failures[record["score"]["failure_category"] or "unknown"] += 1
    n = len(records)
    pass_rate = sum(r["score"]["passed"] for r in records) / n if n else 0.0
    # Unbiased pass@k estimate from n attempts with c successes.
    pass_at_k = []
    avg_at_k = []
    for outcomes in groups.values():
        attempts = len(outcomes)
        sample = min(k, attempts)
        successes = sum(outcomes)
        pass_at_k.append(1 - comb(attempts - successes, sample) / comb(attempts, sample) if sample else 0.0)
        avg_at_k.append(successes / attempts if attempts else 0.0)
    return {
        "runs": n,
        "tasks": len(groups),
        "pass_rate": pass_rate,
        "pass_at_k": sum(pass_at_k) / len(pass_at_k) if pass_at_k else 0.0,
        "avg_at_k": sum(avg_at_k) / len(avg_at_k) if avg_at_k else 0.0,
        "k": k,
        "failures": dict(failures),
    }
