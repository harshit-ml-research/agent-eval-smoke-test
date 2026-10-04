import json
import time
from typing import Any, Callable

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
    max_tokens: int = 256,
    tools: list[dict[str, Any]] | None = None,
    execute_tool: Callable[[str, str], dict[str, Any]] | None = None,
    finalize_on_last_turn: bool = False,
    validate_answer: Callable[[str], Any] | None = None,
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
    available_tools = TOOLS if tools is None else tools
    tool_executor = execute if execute_tool is None else execute_tool
    for turn in range(max_turns):
        final_turn = finalize_on_last_turn and turn == max_turns - 1
        if final_turn:
            messages.append({"role": "user", "content": "Tool budget is exhausted. Return the required compact JSON diagnosis now using observed evidence. State uncertainty when needed."})
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            status = "timeout"
            break
        request = {
            "messages": json.loads(json.dumps(messages)),
            "tools": available_tools if allow_tools and not final_turn else [],
            "temperature": 0,
            "max_tokens": max_tokens,
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
        if result.raw_response.get("finish_reason") == "length":
            status = "truncated_output"
            error = "Generation reached the output token limit"
            break
        if result.raw_response.get("parser_error"):
            status = "malformed_output"
            error = result.raw_response["parser_error"]
            break
        calls = message.get("tool_calls") or []
        if calls:
            if not allow_tools or final_turn:
                status = "unexpected_tool_call"
                break
            for call in calls:
                function = call.get("function") or {}
                observation = tool_executor(function.get("name", ""), function.get("arguments", ""))
                trajectory[-1]["observations"].append({"call": call, "result": observation})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", "missing"),
                    "content": json.dumps({key: value for key, value in observation.items()
                                           if key != "raw_result"}),
                })
            continue
        answer = message.get("content") or ""
        status = "completed" if answer.strip() else "malformed_output"
        if status == "completed" and validate_answer is not None:
            try:
                validate_answer(answer)
            except ValueError as exc:
                status = "invalid_answer"
                error = f"{type(exc).__name__}: {exc}"
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
