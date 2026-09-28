import json
from typing import Any, Callable


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


FUNCTIONS: dict[str, Callable[..., int]] = {"add": add, "multiply": multiply}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Compute the {name} of two integers. Use this tool for arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
    for name in FUNCTIONS
]


def execute(name: str, arguments: str) -> dict[str, Any]:
    try:
        values = json.loads(arguments)
        if name not in FUNCTIONS:
            raise ValueError(f"Unknown tool: {name}")
        if not isinstance(values, dict) or set(values) != {"a", "b"}:
            raise ValueError("Expected exactly a and b")
        if any(type(values[key]) is not int for key in ("a", "b")):
            raise ValueError("a and b must be integers")
        return {"ok": True, "result": FUNCTIONS[name](**values)}
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}
