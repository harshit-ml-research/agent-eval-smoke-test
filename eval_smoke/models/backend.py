from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Generation:
    message: dict[str, Any]
    latency_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class ModelBackend(Protocol):
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str,
        timeout: float,
    ) -> Generation: ...
