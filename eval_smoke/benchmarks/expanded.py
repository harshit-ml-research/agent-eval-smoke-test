"""Deterministic protocol checks with exact tool trace scoring."""

import json
import re

from .base import Task
from .synthetic import SYSTEM_PROMPT


def tasks() -> list[Task]:
    rows = []
    for index, word in enumerate(("blue", "orbit", "copper", "amber", "violet", "harbor", "maple", "quartz"), 1):
        rows.append(Task(f"direct_{index}", f"Reply with exactly: {word}", word, category="direct"))
    for index, word in enumerate(("north", "cedar", "silver", "delta", "marble", "lantern", "comet", "meadow"), 1):
        rows.append(Task(f"no_tool_{index}", f"Tools are available, but do not use them. Reply with exactly: {word}", word, allow_tools=True, category="no_tool"))
    single = (
        ("add", 17, 25), ("multiply", 7, 8), ("add", 39, 46), ("multiply", 9, 12),
        ("add", 58, 27), ("multiply", 13, 6), ("add", 125, 76), ("multiply", 14, 11),
    )
    for index, (name, a, b) in enumerate(single, 1):
        result = a + b if name == "add" else a * b
        rows.append(Task(
            f"single_{index}",
            f"Use the {name} tool with a={a} and b={b}. Reply with only its result.",
            str(result), True, category="single_tool", expected_calls=((name, a, b),),
        ))
    chains = (
        (("add", 8, 9), ("multiply", 17, 3), 51),
        (("multiply", 6, 7), ("add", 42, 8), 50),
        (("add", 23, 19), ("multiply", 42, 4), 168),
        (("multiply", 12, 5), ("add", 60, 17), 77),
        (("add", 64, 28), ("multiply", 92, 2), 184),
        (("multiply", 11, 13), ("add", 143, 29), 172),
        (("add", 37, 48), ("multiply", 85, 6), 510),
        (("multiply", 15, 8), ("add", 120, 35), 155),
    )
    for index, (first, second, result) in enumerate(chains, 1):
        rows.append(Task(
            f"chain_{index}",
            f"Use tools in sequence: {first[0]} a={first[1]} and b={first[2]}, then {second[0]} the first result with {second[2]}. Reply with only the final number.",
            str(result), True, category="chain", expected_calls=(first, second),
        ))
    return rows


def score(task: Task, result: dict) -> dict:
    answer = re.sub(r"\s+", " ", result["answer"].strip()).strip(". ")
    observed = []
    tool_error = False
    for turn in result["trajectory"]:
        for observation in turn.get("observations", []):
            function = observation["call"]["function"]
            try:
                arguments = json.loads(function["arguments"])
                observed.append((function["name"], arguments["a"], arguments["b"]))
            except (ValueError, KeyError, TypeError):
                observed.append((function.get("name"), None, None))
            tool_error |= not observation["result"]["ok"]
    if result["status"] != "completed":
        failure = "empty_output" if result["status"] == "malformed_output" else result["status"]
    elif tool_error:
        failure = "tool_execution_error"
    elif any(observed[index] == observed[index - 1] for index in range(1, len(observed))):
        failure = "repeated_tool_call"
    elif tuple(observed) != task.expected_calls:
        raw = "\n".join(turn.get("response", {}).get("raw_text", "") for turn in result["trajectory"])
        if task.expected_calls and not observed and ("<tool_call" in raw or '"name"' in raw):
            failure = "unparsed_tool_call"
        else:
            failure = "unexpected_tool_call" if not task.expected_calls else "wrong_tool_sequence"
    elif answer.lower() != task.expected.lower():
        failure = "wrong_answer"
    else:
        failure = None
    return {"passed": failure is None, "failure_category": failure}
