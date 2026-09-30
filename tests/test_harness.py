from eval_smoke.analysis.metrics import summarize
from eval_smoke.benchmarks.expanded import score as expanded_score, tasks as expanded_tasks
from eval_smoke.harness.agent import run_agent
from eval_smoke.harness.tools import execute
from eval_smoke.itbench_lite import root_cause_hit
from eval_smoke.models.backend import Generation
from eval_smoke.models.param2 import parse_output


class ScriptedBackend:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        message = next(self.messages)
        return Generation(message, 0.01, {"total_tokens": 2}, {"choices": [{"message": message}]})


def test_tool_chain_and_trajectory():
    backend = ScriptedBackend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "add", "arguments": '{"a":8,"b":9}'}}]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "type": "function", "function": {"name": "multiply", "arguments": '{"a":17,"b":3}'}}]},
        {"role": "assistant", "content": "51"},
    ])
    result = run_agent(backend, "test", "compute", "system", True)
    assert result["status"] == "completed"
    assert result["answer"] == "51"
    assert result["tool_calls"] == 2
    assert result["trajectory"][0]["observations"][0]["result"]["result"] == 17


def test_malformed_and_max_turns():
    assert run_agent(ScriptedBackend([{"role": "assistant", "content": ""}]), "test", "", "", False)["status"] == "malformed_output"
    looping = ScriptedBackend([{"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":1}'}}]}])
    assert run_agent(looping, "test", "", "", True, max_turns=1)["status"] == "max_turns"
    assert execute("add", "bad")["ok"] is False


def test_final_turn_removes_tools_and_requests_diagnosis():
    backend = ScriptedBackend([
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "add", "arguments": '{"a":1,"b":1}'}}]},
        {"role": "assistant", "content": '{"entities":[],"propagations":[],"alerts_explained":[]}'},
    ])
    result = run_agent(backend, "test", "investigate", "system", True, max_turns=2, finalize_on_last_turn=True)
    assert result["status"] == "completed"
    assert result["tool_calls"] == 1
    assert backend.requests[0]["tools"]
    assert backend.requests[1]["tools"] == []
    assert "Tool budget is exhausted" in backend.requests[1]["messages"][-1]["content"]


def test_metrics():
    records = [
        {"task_id": "a", "score": {"passed": True, "failure_category": None}},
        {"task_id": "a", "score": {"passed": False, "failure_category": "incorrect_answer_or_tools"}},
        {"task_id": "b", "score": {"passed": False, "failure_category": "model_error"}},
        {"task_id": "b", "score": {"passed": False, "failure_category": "model_error"}},
    ]
    metrics = summarize(records, k=2)
    assert metrics["pass_rate"] == 0.25
    assert metrics["pass_at_k"] == 0.5
    assert metrics["avg_at_k"] == 0.25


def test_param2_output_parser():
    message = parse_output('<think>work</think>\n<tool_call>\n{"name":"add","arguments":{"a":8,"b":9}}\n</tool_call><<|EOS|>>')
    assert message["reasoning_content"] == "work"
    assert message["content"] == ""
    assert message["tool_calls"][0]["function"] == {"name": "add", "arguments": '{"a": 8, "b": 9}'}
    assert parse_output("<think>work</think>\nblue<<|EOS|>>")["content"] == "blue"


def test_expanded_suite_scores_exact_tool_sequence():
    cases = expanded_tasks()
    assert len(cases) == 32
    assert len({task.id for task in cases}) == 32
    task = next(task for task in cases if task.id == "chain_1")
    result = {
        "status": "completed", "answer": "51",
        "trajectory": [{"observations": [
            {"call": {"function": {"name": "add", "arguments": '{"a":8,"b":9}'}}, "result": {"ok": True}},
            {"call": {"function": {"name": "multiply", "arguments": '{"a":17,"b":3}'}}, "result": {"ok": True}},
        ]}],
    }
    assert expanded_score(task, result)["passed"]
    result["trajectory"][0]["observations"].pop(0)
    assert expanded_score(task, result)["failure_category"] == "wrong_tool_sequence"


def test_itbench_root_cause_proxy_uses_entity_kind_and_namespace():
    ground_truth = {"groups": [
        {"id": "cause", "kind": "Pod", "namespace": "otel-demo", "filter": ["load-generator-.*"], "root_cause": True},
        {"id": "impact", "kind": "Service", "namespace": "otel-demo", "filter": ["frontend"], "root_cause": False},
    ], "aliases": []}
    assert root_cause_hit({"entities": [{"name": "otel-demo/Pod/load-generator-123", "contributing_factor": True}]}, ground_truth)
    assert not root_cause_hit({"entities": [{"name": "otel-demo/Service/frontend", "contributing_factor": True}]}, ground_truth)
