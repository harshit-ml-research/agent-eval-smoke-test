"""Local Transformers backend for the Param2 thinking checkpoint."""

import json
import os
import re
import time
from typing import Any

from .backend import Generation


THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_output(raw: str) -> dict[str, Any]:
    text = raw.split("<<|EOS|>>", 1)[0].strip()
    # Reject partial, reversed, and nested protocol blocks before parsing content.
    active = None
    for tag in re.findall(r"</?(?:think|tool_call)>", text):
        if tag.startswith("</"):
            if active != tag[2:-1]:
                raise ValueError("Unmatched protocol closing tag")
            active = None
        else:
            if active is not None:
                raise ValueError("Nested protocol block")
            active = tag[1:-1]
    if active is not None:
        raise ValueError(f"Unclosed {active} block")
    reasoning = "\n".join(match.strip() for match in THINK.findall(text) if match.strip())
    text = THINK.sub("", text)
    calls = []
    for index, match in enumerate(TOOL_CALL.findall(text), 1):
        call = json.loads(match)
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            raise ValueError("Invalid tool call")
        arguments = call.get("arguments")
        if not isinstance(arguments, (dict, str)):
            raise ValueError("Invalid tool arguments")
        calls.append({
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
            },
        })
    content = TOOL_CALL.sub("", text).strip()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


class Param2TransformersBackend:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        memory_config = os.environ.get("PARAM2_MAX_MEMORY_JSON")
        max_memory = {0: "20GiB", 1: "20GiB", "cpu": "64GiB"}
        if memory_config:
            configured = json.loads(memory_config)
            if not isinstance(configured, dict) or not configured:
                raise ValueError("PARAM2_MAX_MEMORY_JSON must be a nonempty object")
            max_memory = {}
            for key, value in configured.items():
                if key != "cpu" and not key.isdecimal():
                    raise ValueError("Memory keys must be visible GPU indices or cpu")
                if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*(?:GiB|MiB)", value):
                    raise ValueError("Memory values must be positive GiB or MiB amounts")
                max_memory[int(key) if key.isdecimal() else key] = value
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        model: str,
        timeout: float,
    ) -> Generation:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        options: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
            "max_time": timeout,
        }
        if temperature > 0:
            options["temperature"] = temperature
        started = time.monotonic()
        with self.torch.no_grad():
            output = self.model.generate(inputs, **options)
        latency = time.monotonic() - started
        generated = output[0][inputs.shape[-1]:]
        raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        response = {"raw_text": raw, "finish_reason": "length" if len(generated) >= max_tokens else "stop"}
        try:
            message = parse_output(raw)
        except ValueError as exc:
            message = {"role": "assistant", "content": raw}
            response["parser_error"] = f"{type(exc).__name__}: {exc}"
        usage = {
            "prompt_tokens": inputs.shape[-1],
            "completion_tokens": generated.shape[-1],
            "total_tokens": output.shape[-1],
        }
        return Generation(message, latency, usage, response)
