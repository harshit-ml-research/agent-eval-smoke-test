import json
import time
from typing import Any

from eval_smoke.models.backend import ModelBackend

from .tools import TOOLS, execute


def run_agent(
    backend: ModelBackend,
    model: str,
    prompt: str,
    system_prompt: str,
    allow_tools: bool,
    max_turns: int = 5,
    timeout: float = 120,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    trajectory: list[dict[str, Any]] = []
    started = time.monotonic()
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error = None
    status = "max_turns"
    answer = ""
    for turn in range(max_turns):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            status = "timeout"
            break
        request = {
            "messages": json.loads(json.dumps(messages)),
            "tools": TOOLS if allow_tools else [],
            "temperature": 0,
            "max_tokens": 256,
            "model": model,
            "timeout": remaining,
        }
        try:
            result = backend.generate(**request)
        except Exception as exc:
            status = "model_error"
            error = f"{type(exc).__name__}: {exc}"
            trajectory.append({"turn": turn + 1, "request": request, "error": error})
            break
        message = result.message
        trajectory.append({
            "turn": turn + 1,
            "request": request,
            "response": result.raw_response,
            "message": message,
            "latency_seconds": result.latency_seconds,
            "usage": result.usage,
            "observations": [],
        })
        for key in totals:
            totals[key] += result.usage.get(key, 0) or 0
        messages.append(message)
        calls = message.get("tool_calls") or []
        if calls:
            if not allow_tools:
                status = "unexpected_tool_call"
                break
            for call in calls:
                function = call.get("function") or {}
                observation = execute(function.get("name", ""), function.get("arguments", ""))
                trajectory[-1]["observations"].append({"call": call, "result": observation})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "missing"),
                    "content": json.dumps(observation),
                })
            continue
        answer = message.get("content") or ""
        status = "completed" if answer.strip() else "malformed_output"
        break
    return {
        "status": status,
        "answer": answer,
        "messages": messages,
        "trajectory": trajectory,
        "latency_seconds": time.monotonic() - started,
        "usage": totals,
        "error": error,
        "tool_calls": sum(len(item.get("observations", [])) for item in trajectory),
    }
