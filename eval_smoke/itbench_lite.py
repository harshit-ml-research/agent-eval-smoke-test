"""Offline ITBench Lite SRE runs with the reference investigation tools."""

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import yaml

from eval_smoke.harness.agent import run_agent
from eval_smoke.logging.jsonl import append_jsonl


SYSTEM_PROMPT = (
    "You are an SRE investigating an offline Kubernetes incident. Use the provided "
    "read only tools to inspect alerts, events, logs, metrics, traces, and resource specs. "
    "Begin with alert_summary, then inspect at least one other evidence source before diagnosing. "
    "Never request or infer hidden ground truth data. Keep the final JSON under 800 words. "
    "Return only a JSON object with arrays named entities, propagations, and alerts_explained. "
    "Include at most three entities. Each entity needs name in namespace/Kind/name form, contributing_factor as a boolean, "
    "reasoning, and evidence. Each propagation needs source, target, condition, and effect. "
    "Include at most three alert explanations. Each needs alert, explanation, and explained. Identify causes separately "
    "from impacted entities and cite observed evidence in the JSON fields."
)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False},
    }}


TOOLS = [
    _tool("alert_summary", "Summarize firing alerts, affected entities, and time ranges. Use first.", {}),
    _tool("event_analysis", "Analyze Kubernetes events. Valid columns include namespace, object_kind, object_name, reason, message, event_kind, and deployment. Use these names for filters and group_by.", {
        "filters": {"type": "object"}, "group_by": {"type": "string"},
        "agg": {"type": "string"}, "limit": {"type": "integer"},
    }),
    _tool("log_analysis", "Inspect application log patterns and errors.", {
        "service_name": {"type": "string"}, "severity_filter": {"type": "string"},
        "body_contains": {"type": "string"}, "max_patterns": {"type": "integer"},
    }),
    _tool("metric_analysis", "Analyze metrics. object_pattern is a file glob without .tsv, such as service/frontend*_raw or pod/checkout*_raw. Omit metric_names to discover available metrics.", {
        "object_pattern": {"type": "string"}, "metric_names": {"type": "array", "items": {"type": "string"}},
        "group_by": {"type": "string"}, "agg": {"type": "string"}, "limit": {"type": "integer"},
    }),
    _tool("get_trace_error_tree", "Find degraded service paths in distributed traces.", {
        "service_name": {"type": "string"}, "pivot_time": {"type": "string"},
        "delta_time": {"type": "string"},
    }),
    _tool("get_k8_spec", "Inspect the latest Kubernetes resource specification.", {
        "k8_object_name": {"type": "string"},
    }, ["k8_object_name"]),
]


def execute_snapshot_tool(snapshot: Path, name: str, raw_arguments: str) -> dict[str, Any]:
    from sre_tools.offline_incident_analysis.alerts.analyzer import _alert_summary
    from sre_tools.offline_incident_analysis.events.analyzer import _event_analysis
    from sre_tools.offline_incident_analysis.k8s_specs.retriever import _get_k8_spec
    from sre_tools.offline_incident_analysis.logs.analyzer import _log_analysis
    from sre_tools.offline_incident_analysis.metrics.analyzer import _metric_analysis
    from sre_tools.offline_incident_analysis.traces.analyzer import _get_trace_error_tree

    handlers = {
        "alert_summary": (_alert_summary, {"base_dir": str(snapshot), "limit": 30}),
        "event_analysis": (_event_analysis, {"events_file": str(snapshot / "k8s_events_raw.tsv"), "limit": 50}),
        "log_analysis": (_log_analysis, {"logs_file": str(snapshot / "otel_logs_raw.tsv"), "max_patterns": 20}),
        "metric_analysis": (_metric_analysis, {"base_dir": str(snapshot / "metrics"), "limit": 50}),
        "get_trace_error_tree": (_get_trace_error_tree, {"trace_file": str(snapshot / "otel_traces_raw.tsv")}),
        "get_k8_spec": (_get_k8_spec, {"k8s_objects_file": str(snapshot / "k8s_objects_raw.tsv")}),
    }
    if name not in handlers:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    try:
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        handler, fixed = handlers[name]
        parts = asyncio.run(handler({**arguments, **fixed}))
        output = "\n".join(part.text for part in parts)
        if len(output) > 16000:
            output = output[:16000] + "\n[tool output truncated]"
        return {"ok": True, "result": output}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def root_cause_hit(diagnosis: dict[str, Any], ground_truth: dict[str, Any]) -> bool:
    groups = ground_truth.get("groups", [])
    root_ids = {group["id"] for group in groups if group.get("root_cause")}
    for aliases in ground_truth.get("aliases", []):
        if root_ids.intersection(aliases):
            root_ids.update(aliases)
    for entity in diagnosis.get("entities", []):
        if not isinstance(entity, dict) or entity.get("contributing_factor") is not True:
            continue
        name = entity.get("name", "")
        parts = name.split("/", 2)
        if len(parts) != 3:
            continue
        namespace, kind, resource = parts
        for group in groups:
            if group["id"] not in root_ids or group.get("namespace") != namespace or group.get("kind", "").lower() != kind.lower():
                continue
            if any(re.search(pattern, resource) for pattern in group.get("filter", [])):
                return True
    return False


def run_scenarios(backend: Any, model: str, snapshot_root: Path, output: Path, limit: int, max_tokens: int, max_turns: int, timeout: int) -> dict[str, Any]:
    scenarios = sorted(snapshot_root.glob("Scenario-*"), key=lambda path: int(path.name.split("-")[-1]))[:limit]
    if len(scenarios) != limit:
        raise ValueError(f"Expected {limit} scenarios in {snapshot_root}, found {len(scenarios)}")
    records = []
    for snapshot in scenarios:
        result = run_agent(
            backend, model, "Investigate this incident. Start with alert_summary, then use other evidence as needed. Return the required JSON diagnosis.",
            SYSTEM_PROMPT, True, max_turns=max_turns, timeout=timeout, max_tokens=max_tokens,
            tools=TOOLS, execute_tool=lambda name, arguments: execute_snapshot_tool(snapshot, name, arguments),
            finalize_on_last_turn=True,
        )
        try:
            diagnosis = json.loads(result["answer"])
            if not isinstance(diagnosis, dict) or any(not isinstance(diagnosis.get(key), list) for key in ("entities", "propagations", "alerts_explained")):
                raise ValueError("Missing diagnosis arrays")
            parse_error = None
        except (json.JSONDecodeError, ValueError) as exc:
            diagnosis = None
            parse_error = f"{type(exc).__name__}: {exc}"
        ground_truth = yaml.safe_load((snapshot / "ground_truth.yaml").read_text())
        hit = root_cause_hit(diagnosis, ground_truth) if diagnosis else False
        if diagnosis:
            destination = output.parent / "agent_outputs" / snapshot.name / "1" / "agent_output.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n")
        record = {"scenario": snapshot.name, "model": model, "root_cause_hit_proxy": hit, "parse_error": parse_error, **result}
        append_jsonl(output, record)
        records.append(record)
        print(f"{snapshot.name}: root_cause_hit_proxy={hit}, status={result['status']}, tool_calls={result['tool_calls']}", flush=True)
    metrics = {
        "scenarios": len(records), "root_cause_hit_proxy": sum(row["root_cause_hit_proxy"] for row in records) / len(records),
        "valid_json": sum(row["parse_error"] is None for row in records),
        "tool_calls": sum(row["tool_calls"] for row in records),
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("param2", "openai"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-turns", type=int, default=9)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.max_turns < 1:
        parser.error("--max-turns must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")
    if args.backend == "param2":
        from eval_smoke.models.param2 import Param2TransformersBackend
        backend = Param2TransformersBackend(args.model)
    else:
        from eval_smoke.models.granite import OpenAICompatibleBackend
        backend = OpenAICompatibleBackend(args.base_url, reasoning_effort="none")
    print(json.dumps(run_scenarios(backend, args.model, args.snapshot_root, args.output, args.limit, args.max_tokens, args.max_turns, args.timeout), indent=2))


if __name__ == "__main__":
    main()
