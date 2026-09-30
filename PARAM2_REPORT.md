# Param2 agent harness run

Run date: September 30, 2026. Host: `passpoli`. Model: `bharatgenai/Param2-17B-A2.4B-Thinking`, revision `86fd8f7bb96a404aaab589cb56831f575bbb92b5`. The local Transformers backend used PyTorch 2.6.0 and Transformers 4.52.3 in `~/research` across both RTX A5000 GPUs. Decoding was deterministic, with a 512 token output cap and a 600 second task timeout.

The eight synthetic tasks completed. Five passed. The JSONL trajectory is at `results/param2-20260930T211030.jsonl`, with aggregate metrics at `results/param2-20260930T211030.metrics.json` in this project and on the server. The run recorded 4,842 tokens, three executed tool calls, and 341.84 seconds of task latency. Checkpoint loading took about five minutes before task generation. GPU memory during generation was 14,582 MiB on GPU 0 and 19,902 MiB on GPU 1. Both returned to 2 MiB after the process exited.

| Model and run | Passed | Failed | Observed tool behavior |
| --- | ---: | ---: | --- |
| Granite 3B, September 29 | 6/8 | 2/8 | Two single tool prompts returned empty output. Both chains passed. |
| Param2 17B, September 30 | 5/8 | 3/8 | One chain passed with two calls. Two single tool calls used unsupported output forms. One chain skipped a required call. |

## Param2 task results

| Task | Result | Observation |
| --- | --- | --- |
| `plain_1`, `plain_2`, `plain_3` | Pass | Exact requested words returned. |
| `one_tool_1` | Fail | Model emitted an untagged JSON call for `add`; the harness treated it as answer text and executed no tool. |
| `one_tool_2` | Fail | Model emitted a self closing XML style `multiply` call; the harness executed no tool. |
| `multi_tool_1` | Pass | Model called `add`, then `multiply`, and returned `51`. |
| `multi_tool_2` | Fail | Model mentally computed `6 * 7`, called only `add`, and returned the correct number `50`. The task required both tools. |
| `malformed_1` | Pass | The task deliberately asked for an empty response. The empty final answer was correctly classified as malformed output. This pass does not indicate useful task completion. |

## Failure hypotheses from the team

This run directly shows inconsistent tool call formatting and a skipped required call. The two unsupported forms do not match the checkpoint's `<tool_call>` JSON chat template, so the current parser did not execute them. The trace cannot establish whether either form would persist under different prompting or serving software.

No repeated calls or context exhaustion appeared in these eight tasks. Empty final output appeared only in the task that explicitly requested it. The sample is a harness validation, not a broad estimate of agent benchmark performance. Pass rate and pass@1 are both 0.625 because each task had one attempt. The later ITBench Lite run is documented in [ITBENCH_LITE_REPORT.md](ITBENCH_LITE_REPORT.md).

## Reproduction

```bash
ssh harshitsingh@10.119.2.11 'cd ~/professional-agent-evals/agent-eval-smoke-test && bash scripts/run_param2.sh'
```

The script checks both GPUs before loading the model and saves a new timestamped trajectory and metrics file.
