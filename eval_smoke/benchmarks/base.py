from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    expected: str
    allow_tools: bool = False
    min_tool_calls: int = 0
    expect_error: bool = False
    category: str = ""
    expected_calls: tuple[tuple[str, int, int], ...] = ()


class BenchmarkAdapter(Protocol):
    def tasks(self) -> list[Task]: ...

    def score(self, task: Task, result: dict) -> dict: ...
