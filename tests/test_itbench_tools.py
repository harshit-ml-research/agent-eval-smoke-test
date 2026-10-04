"""Regression coverage for incident evidence selection and tool failures."""

import json

import pytest
from jsonschema import Draft202012Validator

from eval_smoke.itbench_lite import TOOLS, execute_snapshot_tool
from eval_smoke.itbench_lite import run_scenarios
from eval_smoke.models.backend import Generation


def test_event_schema_accepts_scalar_filters_and_multiple_group_columns():
    schema = next(tool['function']['parameters'] for tool in TOOLS
                  if tool['function']['name'] == 'event_analysis')
    validator = Draft202012Validator(schema)
    validator.validate({'filters': {'namespace': 'otel-demo', 'object_kind': 'Pod'},
                        'group_by': ['namespace', 'object_name'], 'agg': 'count'})
    assert list(validator.iter_errors({'filters': {'object_kind': ['Pod', 'Deployment']}}))
    assert list(validator.iter_errors({'filters': {'alertname': 'RequestErrorRate'}}))
    assert list(validator.iter_errors({'group_by': 'entity'}))


@pytest.mark.parametrize('arguments', ['bad json', '[]', '{"filters":{"object_kind":["Pod","Deployment"]}}'])
def test_invalid_event_arguments_are_rejected_before_execution(tmp_path, arguments):
    result = execute_snapshot_tool(tmp_path, 'event_analysis', arguments)
    assert result['ok'] is False
    assert result['error_type'] == 'invalid_arguments'


def test_suite_counts_tool_failures_and_preserves_observations(tmp_path):
    snapshot = tmp_path / 'snapshots' / 'Scenario-1'
    snapshot.mkdir(parents=True)
    (snapshot / 'ground_truth.yaml').write_text('groups: []\n')
    messages = iter([
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_1', 'type': 'function', 'function': {
                'name': 'event_analysis', 'arguments': '{"group_by":"entity"}'}}]},
        {'role': 'assistant', 'content': '{"entities":[],"propagations":[],"alerts_explained":[]}'},
    ])

    class Backend:
        def generate(self, **kwargs):
            return Generation(next(messages), 0.01)

    output = tmp_path / 'trajectory.jsonl'
    metrics = run_scenarios(Backend(), 'test', snapshot.parent, output, 1, 100, 2, 60)
    assert metrics['tool_calls'] == metrics['tool_errors'] == 1
    record = json.loads(output.read_text())
    assert record['trajectory'][0]['observations'][0]['result']['error_type'] == 'invalid_arguments'


@pytest.fixture
def reference_tools():
    pytest.importorskip('pandas')
    pytest.importorskip('mcp')
    pytest.importorskip('sre_tools')


@pytest.mark.parametrize('nested', [True, False])
def test_alerts_use_incident_evidence_not_operational_json(tmp_path, reference_tools, nested):
    directory = tmp_path / 'alerts' if nested else tmp_path
    directory.mkdir(exist_ok=True)
    alert = {'labels': {'alertname': 'RequestErrorRate', 'namespace': 'otel-demo',
                        'service_name': 'checkout', 'severity': 'warning'}, 'state': 'firing'}
    (directory / 'alerts_at_2025-12-15T17-30-00.json').write_text(json.dumps([alert]))
    if nested:
        (tmp_path / 'status.json').write_text('{"events":[]}')
        (tmp_path / 'assertion.json').write_text('{"status":{}}')
    result = execute_snapshot_tool(tmp_path, 'alert_summary', '{}')
    assert result['ok'] is True
    summaries = json.loads(result['result'])
    assert [row['alertname'] for row in summaries] == ['RequestErrorRate']
    assert summaries[0]['entity'] == 'checkout'


def test_valid_event_filters_reach_reference_tool(tmp_path, reference_tools):
    (tmp_path / 'k8s_events_raw.tsv').write_text(
        'namespace\tobject_kind\tobject_name\treason\tevent_time\n'
        'otel-demo\tPod\tcheckout-123\tBackOff\t2025-12-15T17:30:00Z\n'
        'otel-demo\tService\tfrontend\tCreated\t2025-12-15T17:30:00Z\n')
    result = execute_snapshot_tool(tmp_path, 'event_analysis', json.dumps({
        'filters': {'namespace': 'otel-demo', 'object_kind': 'Pod'},
        'group_by': ['namespace', 'object_kind'], 'agg': 'count'}))
    assert result['ok'] is True
    assert json.loads(result['result'])['data'] == [
        {'namespace': 'otel-demo', 'object_kind': 'Pod', 'count': 1}]


@pytest.mark.parametrize('group_columns', [
    ['namespace', 'reason'], ['namespace', 'message'], ['namespace', 'event_kind'],
    ['reason', 'message', 'event_kind'], ['reason', 'message', 'event_kind', 'count'],
])
@pytest.mark.parametrize('empty', [False, True])
def test_event_list_aggregation_does_not_duplicate_group_keys(tmp_path, reference_tools, group_columns, empty):
    (tmp_path / 'k8s_events_raw.tsv').write_text(
        'namespace\tobject_kind\tobject_name\treason\tmessage\tevent_kind\tcount\tevent_time\n'
        'otel-demo\tPod\tcheckout-123\tBackOff\tConnection refused\tWarning\t1\t2025-12-15T17:30:00Z\n')
    result = execute_snapshot_tool(tmp_path, 'event_analysis', json.dumps({
        'filters': {'namespace': 'missing' if empty else 'otel-demo'},
        'group_by': group_columns, 'agg': 'list'}))
    assert result['ok'] is True, result
    payload = json.loads(result['result'])
    assert payload['returned_count'] == (0 if empty else 1)
    if not empty:
        for column in group_columns:
            assert not isinstance(payload['data'][0][column], list)


@pytest.mark.parametrize('name,module_name,handler_name,key,default', [
    ('event_analysis', 'events', '_event_analysis', 'limit', 50),
    ('log_analysis', 'logs', '_log_analysis', 'max_patterns', 20),
    ('metric_analysis', 'metrics', '_metric_analysis', 'limit', 50),
])
def test_output_controls_honor_requests_and_retain_defaults(tmp_path, reference_tools, monkeypatch,
                                                          name, module_name, handler_name, key, default):
    import importlib
    from mcp.types import TextContent

    module = importlib.import_module(f'sre_tools.offline_incident_analysis.{module_name}.analyzer')
    requests = []

    async def capture(arguments):
        requests.append(arguments)
        return [TextContent(type='text', text='[]')]

    monkeypatch.setattr(module, handler_name, capture)
    assert execute_snapshot_tool(tmp_path, name, json.dumps({key: 1}))['ok'] is True
    assert execute_snapshot_tool(tmp_path, name, '{}')['ok'] is True
    assert [request[key] for request in requests] == [1, default]
    protected_key = {'event_analysis': 'events_file', 'log_analysis': 'logs_file',
                     'metric_analysis': 'base_dir'}[name]
    rejected = execute_snapshot_tool(tmp_path, name, json.dumps({protected_key: '/outside'}))
    assert rejected['error_type'] == 'invalid_arguments'
    assert len(requests) == 2


def test_full_evidence_is_saved_without_entering_model_context(tmp_path, reference_tools, monkeypatch):
    from mcp.types import TextContent
    from eval_smoke.harness.agent import run_agent
    from sre_tools.offline_incident_analysis.events import analyzer

    full_output = json.dumps({'data': [{'message': 'x' * 17000}]})

    async def response(arguments):
        return [TextContent(type='text', text=full_output)]

    monkeypatch.setattr(analyzer, '_event_analysis', response)
    requests = []

    class Backend:
        def generate(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                message = {'role': 'assistant', 'tool_calls': [{'id': 'call_1',
                           'function': {'name': 'event_analysis', 'arguments': '{}'}}]}
            else:
                message = {'role': 'assistant', 'content': 'done'}
            return Generation(message, 0.01)

    result = run_agent(Backend(), 'test', 'investigate', 'system', True, max_turns=2,
                       tools=TOOLS, execute_tool=lambda name, arguments: execute_snapshot_tool(tmp_path, name, arguments))
    saved = result['trajectory'][0]['observations'][0]['result']
    model_visible = json.loads(requests[1]['messages'][-1]['content'])
    assert saved['raw_result'] == full_output
    assert saved['result'].endswith('[tool output truncated]')
    assert 'raw_result' not in model_visible
    assert model_visible == {key: value for key, value in saved.items() if key != 'raw_result'}
    assert full_output not in requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('response', [
    'Error: Group column entity not found',
    'Error reading events file: missing',
    'No metric files found matching pattern',
    '{"error":"No objects matching checkout found"}',
])
def test_returned_tool_errors_are_not_successes(tmp_path, reference_tools, monkeypatch, response):
    from mcp.types import TextContent
    from sre_tools.offline_incident_analysis.alerts import analyzer

    async def fail(arguments):
        return [TextContent(type='text', text=response)]

    monkeypatch.setattr(analyzer, '_alert_summary', fail)
    result = execute_snapshot_tool(tmp_path, 'alert_summary', '{}')
    assert result['ok'] is False
    assert result['error_type'] == 'tool_error'
    assert result['result'] == response
