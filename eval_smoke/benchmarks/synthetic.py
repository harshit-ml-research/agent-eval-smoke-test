import re

from .base import Task


SYSTEM_PROMPT = (
    "You are a precise assistant. Answer with only the requested value. "
    "When arithmetic tools are available, use them for arithmetic. "
    "If a tool fails, correct the call if possible."
)


def tasks() -> list[Task]:
    return [
        Task("plain_1", "Reply with exactly: blue", "blue"),
        Task("plain_2", "Reply with exactly: orbit", "orbit"),
        Task("plain_3", "Reply with exactly: copper", "copper"),
        Task("one_tool_1", "Use the add tool to compute 17 + 25. Reply with only the number.", "42", True, 1),
        Task("one_tool_2", "Use the multiply tool to compute 7 times 8. Reply with only the number.", "56", True, 1),
        Task("multi_tool_1", "Use tools in sequence: add 8 and 9, then multiply the result by 3. Reply with only the number.", "51", True, 2),
        Task("multi_tool_2", "Use tools in sequence: multiply 6 and 7, then add 8. Reply with only the number.", "50", True, 2),
        Task("malformed_1", "Reply with an empty response.", "", False, 0, True),
    ]


def score(task: Task, result: dict) -> dict:
    if task.expect_error:
        passed = result["status"] in {"malformed_output", "model_error", "timeout"}
    else:
        normalized = re.sub(r"\s+", " ", result["answer"].strip()).strip(". ")
        passed = result["status"] == "completed" and normalized.lower() == task.expected.lower()
        passed = passed and result["tool_calls"] >= task.min_tool_calls
    return {"passed": passed, "failure_category": None if passed else result["status"] if result["status"] != "completed" else "incorrect_answer_or_tools"}
