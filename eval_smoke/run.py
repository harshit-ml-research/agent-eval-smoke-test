import argparse
import json
from pathlib import Path

from eval_smoke.harness.controller import run_suite
from eval_smoke.models.granite import OpenAICompatibleBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="granite4.2:3b")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--output", type=Path, default=Path("results/smoke.jsonl"))
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--backend", choices=("openai", "param2"), default="openai")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--suite", choices=("smoke", "expanded"), default="smoke")
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be at least 1")
    if args.backend == "param2":
        from eval_smoke.models.param2 import Param2TransformersBackend
        backend = Param2TransformersBackend(args.model)
    else:
        backend = OpenAICompatibleBackend(args.base_url, args.api_key)
    metrics = run_suite(backend, args.model, args.output, args.attempts, args.timeout, args.max_tokens, args.suite)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
