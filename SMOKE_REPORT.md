# First smoke run

Run date: September 28, 2026. Host: `passpoli` (`harshitsingh@10.119.2.11`). Project path: `/home/harshitsingh/professional-agent-evals/agent-eval-smoke-test`.

The run used `/home/harshitsingh/research/bin/python` (Python 3.11.15). A separate Ollama 0.33.1 service listened on `127.0.0.1:12177` and used model storage under `/home/harshitsingh/professional-agent-evals/ollama-models`. It was restricted to GPU 0 and stopped after the run. The model was `granite4.2:3b`.

Before the run, GPU 0 had 12,726 MiB used and 11,516 MiB free. GPU 1 had 18,746 MiB used and 5,495 MiB free. During the run, GPU 0 peaked at 23,198 MiB used, leaving about 1,043 MiB free. GPU 1 peaked at 18,938 MiB used. The other users' processes were left untouched.

Eight tasks ran. Five passed. The three plain answer tasks passed. The chained addition then multiplication task passed and recorded two tool calls and observations. The two single tool tasks and the other chained task returned empty model output without a tool call. The malformed output task correctly detected and scored an empty response. This establishes that tool calling works, while Granite's behavior on these prompts is inconsistent.

The isolated rerun script was then verified using an Ollama runtime copied under the same account and a 4,096 token context. Six of eight tasks passed. Both chained arithmetic tasks made two tool calls and passed. The two single tool tasks still returned empty output. The script stopped its own service after completion, and GPU 0 returned to 12,726 MiB used. This second run is saved as `results/smoke-20260928T135427.jsonl` with matching metrics.

Complete request and response trajectories, observations, latency, token usage, errors, and scores are saved in `results/smoke.jsonl` on the server. Aggregate metrics are in `results/smoke.metrics.json`. The first run recorded 4,402 total tokens. Pass rate, pass@1, and avg@1 were each 0.625. Unit tests verified pass@2 and avg@2 on a two attempt fixture; all three tests passed.

Rerun with:

```bash
ssh harshitsingh@10.119.2.11 'cd ~/professional-agent-evals/agent-eval-smoke-test && bash scripts/run_smoke.sh'
```

The repo is designed so a vLLM OpenAI compatible endpoint and either future model can be selected with `--base-url` and `--model`, without changing the benchmark or agent loop. Full ITBench integration remains unimplemented by design.
