import json
from pathlib import Path

import pytest

from eval_smoke.harness.agent import run_agent
from eval_smoke.itbench_lite import parse_diagnosis
from eval_smoke.models.backend import Generation
from eval_smoke.models.param2 import parse_output


@pytest.mark.parametrize("raw", ["<think>unfinished", '<tool_call>{"name":"add","arguments":{}}', "</think>bad", "<think><think>x</think></think>"])
def test_incomplete_protocol_is_rejected(raw):
    with pytest.raises(ValueError):
        parse_output(raw)


@pytest.mark.parametrize("finish,parser_error,status", [("length", "partial", "truncated_output"), ("stop", "partial", "malformed_output")])
def test_failure_does_not_execute_partially_generated_calls(finish, parser_error, status):
    class Backend:
        def generate(self, **kwargs):
            return Generation({"role": "assistant", "tool_calls": [{"function": {"name": "add", "arguments": "{}"}}]}, 0, raw_response={"finish_reason": finish, "parser_error": parser_error})
    result = run_agent(Backend(), "test", "", "", True, execute_tool=lambda *args: pytest.fail("Partial output executed"))
    assert result["status"] == status
    assert result["tool_calls"] == 0
    assert result["trajectory"][0]["response"]["parser_error"] == parser_error


def test_diagnosis_contract():
    valid = {"entities": [{"name": "otel-demo/Pod/frontend-123", "contributing_factor": True, "reasoning": "Observed failure", "evidence": ["Error event"]}], "propagations": [], "alerts_explained": [{"alert": "RequestErrorRate", "explanation": "Observed errors", "explained": True}]}
    assert parse_diagnosis(json.dumps(valid)) == valid
    for change in [{"name": "frontend"}, {"contributing_factor": "true"}, {"evidence": ""}]:
        broken = json.loads(json.dumps(valid))
        broken["entities"][0].update(change)
        with pytest.raises(ValueError):
            parse_diagnosis(json.dumps(broken))
    valid["alerts_explained"][0]["explained"] = "yes"
    with pytest.raises(ValueError):
        parse_diagnosis(json.dumps(valid))


def test_invalid_diagnosis_is_not_completed():
    class Backend:
        def generate(self, **kwargs):
            return Generation({"role": "assistant", "content": '{"entities":[{"name":"frontend"}],"propagations":[],"alerts_explained":[]}'}, 0)
    result = run_agent(Backend(), "test", "", "", False, validate_answer=parse_diagnosis)
    assert result["status"] == "invalid_answer"
    assert result["answer"]
