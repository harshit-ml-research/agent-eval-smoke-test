"""OpenAI compatible backend, configured for Granite by default."""

import time
from typing import Any

from openai import OpenAI

from .backend import Generation


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str = "local") -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str,
        timeout: float,
    ) -> Generation:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if tools:
            request["tools"] = tools
        started = time.monotonic()
        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message.model_dump(exclude_none=True)
        return Generation(
            message=message,
            latency_seconds=time.monotonic() - started,
            usage=response.usage.model_dump(exclude_none=True) if response.usage else {},
            raw_response=response.model_dump(exclude_none=True),
        )
