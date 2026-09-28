from eval_smoke.analysis.metrics import summarize
from eval_smoke.harness.agent import run_agent
from eval_smoke.harness.tools import execute
from eval_smoke.models.backend import Generation


class ScriptedBackend:
    def __init__(self, messages):
        self.messages = iter(messages)

    def generate(self, **kwargs):
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
