"""Local Transformers backend for the Param2 thinking checkpoint."""

import json
import re
import time
from typing import Any

from .backend import Generation


THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_output(raw: str) -> dict[str, Any]:
    text = raw.split("<<|EOS|>>", 1)[0].strip()
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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "20GiB", 1: "20GiB", "cpu": "64GiB"},
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
        message = parse_output(raw)
        usage = {
            "prompt_tokens": inputs.shape[-1],
            "completion_tokens": generated.shape[-1],
            "total_tokens": output.shape[-1],
        }
        return Generation(message, latency, usage, {"raw_text": raw})
