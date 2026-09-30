# Agent evaluation report: September 30 to October 1, 2026

This report covers the work started on September 30 and completed after midnight on October 1, India time. All runs used `passpoli` under the `harshitsingh` account. The final comparison is ten [ITBench Lite](https://huggingface.co/datasets/ibm-research/ITBench-Lite) SRE snapshots with a deterministic root cause entity proxy. Raw JSONL trajectories are stored locally in `results/` and in the matching project directory on `passpoli`. That directory is ignored by Git; the committed reports contain the result tables and run identifiers.

## Executive result

| Test | Param2 17B A2.4B Thinking | Granite 4.2 3B | Interpretation |
| --- | ---: | ---: | --- |
| Eight synthetic smoke tasks | 5/8 | 6/8, prior baseline from September 29 | Harness and tool protocol check |
| Expanded synthetic tasks | 7/7 direct tasks before interruption | 25/32 complete | No paired 32 task score |
| Ten ITBench Lite SRE snapshots | 0/10 root cause hits | 0/10 root cause hits | Exact entity proxy, not official judge |
| ITBench minimal JSON schema | 10/10 | 2/10 | Does not measure factual accuracy |
| ITBench tool calls | 0 | 80 | Param2 never observed the incidents |

The main finding is sharper than the equal 0/10 score suggests. Param2 generated the exact same answer for all ten incidents without calling a tool. It invented a checkout pod and cited tools it had not used. Granite called tools eight times per incident but usually ended with an invalid final response. Both failed the root cause proxy through different mechanisms.

The eight task smoke score includes one deliberately malformed task that passes when the model returns an empty response. Excluding that protocol check, Param2 completed 4/7 ordinary tasks. The prior Granite baseline completed 5/7 ordinary tasks.

## Models, data, and scoring

Param2 used `bharatgenai/Param2-17B-A2.4B-Thinking` at revision `86fd8f7bb96a404aaab589cb56831f575bbb92b5`, loaded with PyTorch 2.6.0 and Transformers 4.52.3 across two RTX A5000 GPUs. The ITBench run enabled the model generation cache. Granite used `granite4.2:3b` through a private Ollama instance on one GPU. Granite reasoning was disabled for the ITBench run because earlier probes exhausted output tokens in internal reasoning.

The ITBench Lite dataset was pinned to revision `d0916b08ba421ce5e672e9ad68aa947d938dfef0`. The ten SRE scenarios were 1, 2, 4, 5, 6, 7, 8, 9, 11, and 12. Scenario numbers 3 and 10 are absent at this revision. Both models received the same prompt, six read only tools from the [reference agent](https://github.com/itbench-hub/ITBench-CISO-SRE-FinOps-Agent), a 2,048 output token cap, eight investigation turns, a final turn with tools removed, temperature zero, and a 600 second incident limit. Snapshot paths were fixed by the harness. Ground truth was opened only for scoring after generation.

The score is one binary hit per scenario. The output must contain an entity marked `contributing_factor: true` whose `namespace/Kind/name` matches a ground truth root cause group, including its name pattern and aliases. The [official SRE evaluator](https://github.com/itbench-hub/ITBench-Evaluations) uses a separate LLM judge. No judge credentials were configured, so these numbers are not official ITBench scores. The harness exposes six reference tools, not the complete reference agent environment.

## Chronological run inventory

| Run and artifact | Scope | Result | Decision or finding |
| --- | --- | --- | --- |
| `param2-20260930T211030.jsonl` | Eight synthetic tasks | 5/8, three tool calls | Plain answers passed. Two single tool tasks used unsupported call forms. One chain skipped a required call. |
| `granite-expanded-20260930T222325.jsonl` | 32 expanded synthetic tasks | 25/32, 15 tool calls | Eight direct and eight no tool tasks passed. Single tool tasks passed 3/8; chains passed 6/8. All seven failures were empty outputs with zero calls. |
| `param2-expanded-20260930T222520.jsonl` | Expanded synthetic run, interrupted | First 7 direct tasks passed, zero calls | Stopped when ITBench Lite was selected. It is not a 32 task result or a paired comparison. |
| `itbench-lite-granite-20260930T224819` | Two SRE probes | 0 hits, 0 valid diagnoses, 7 calls | Initial tool and output format check. |
| `itbench-lite-granite-20260930T225346` | One SRE probe | 0 hits, 0 valid diagnoses, 5 calls | Output formatting remained unreliable. |
| `itbench-lite-granite-20260930T225837` | One SRE probe | 0 hits, 0 valid diagnoses, 8 calls | Reached call limit. |
| `itbench-lite-granite-20260930T230034` | One SRE probe | 0 hits, 0 valid diagnoses, 16 calls | Investigation continued without a final answer. |
| `itbench-lite-param2-20260930T230153` | One SRE probe | 0 hits, 0 valid diagnoses, 16 calls | Repeated `alert_summary` after a harness rejection of an extra argument. The reference tool would ignore it, so this is not valid evidence of a model loop. |
| `itbench-lite-param2-20260930T232937` | One SRE probe after wrapper fix | 0 hits, 0 valid diagnoses, 1 call | Tool result was delivered. Param2 then wrote an oversized alert based answer, cut off by the 1,200 second limit. |
| `itbench-lite-granite-20260930T235708` | Ten SRE probes with 16 calls allowed | 0 hits, 0 valid diagnoses, 160 calls | Every incident exhausted the call budget. Tool descriptions were too vague for valid event columns and metric file patterns. Superseded by the final configuration. |
| `itbench-lite-granite-20261001T000723` | One SRE probe with corrected descriptions and forced final turn | 0 hits, 0 valid diagnoses, 8 calls | Emitted a textual tool call after tools were removed. |
| `itbench-lite-param2-20261001T000832` | One SRE probe with generation cache | 0 hits, 1 valid JSON, 0 calls | Faster generation, but the answer invented incident evidence. |
| `itbench-lite-granite-20261001T001649` | Final ten SRE scenarios | 0/10 hits, 2/10 valid diagnoses, 80 calls | Complete Granite result. |
| `itbench-lite-param2-20261001T002109` | Final ten SRE scenarios | 0/10 hits, 10/10 valid JSON, 0 calls | Complete Param2 result. |

One Param2 probe was rejected by the GPU capacity gate and wrote no trajectory. Earlier interrupted or empty ITBench directories also do not contribute scores. The GPU gate prevented the new model load when either GPU lacked capacity; the completed Param2 run used both GPUs. Granite ran on GPU 1 while another process reserved GPU 0, then stopped its private Ollama service after completion.

## Final ITBench scenario results

| Scenario | Param2 root hit | Param2 calls | Granite root hit | Granite calls | Granite final failure |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | 8 | Extra JSON data |
| 2 | 0 | 0 | 0 | 8 | Textual tool call |
| 4 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 5 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 6 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 7 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 8 | 0 | 0 | 0 | 8 | Valid JSON, wrong root entity |
| 9 | 0 | 0 | 0 | 8 | Extra JSON data |
| 11 | 0 | 0 | 0 | 8 | Textual tool call |
| 12 | 0 | 0 | 0 | 8 | Valid JSON, empty diagnosis |

Median incident latency, excluding model loading, was 158.2 seconds for Param2 and 17.1 seconds for Granite. Param2's ten incident latencies totaled 1,564.0 seconds; Granite's totaled 212.3 seconds. Checkpoint loading added roughly four to five minutes to the Param2 run. These times describe this serving setup, not intrinsic model speed.

## Why each model failed

Param2's reasoning planned to call `alert_summary` and other tools, but its raw generation contained no `<tool_call>` marker. It immediately returned final JSON. All ten final answers were byte identical, with 1,409 generated tokens each, because the prompt was identical and it consumed no scenario observations. The answer repeated a fabricated checkout pod three times, cited `alert_summary` and `event_analysis` despite no calls, invented a Kubernetes spec, used names outside `namespace/Kind/name`, and set all `contributing_factor` fields to false. This is evidence skipping followed by fabricated evidence, not a scorer mismatch.

Granite used the tools but spent calls on weak or invalid queries. Early trajectories show nonexistent event columns such as `service_type` and `service`, then corrected queries after the tool returned available columns. It also guessed metric file patterns that returned no files. The corrected tool descriptions reduced those avoidable errors, but every final run still consumed the eight call budget. The final response failed in three ways: six malformed structures, two textual tool calls without tool access, and two structurally valid but incorrect or empty diagnoses. The original 16 call run had no valid final diagnoses and is excluded from the final score.

The eight task smoke run did not show the same Param2 behavior. It completed one two tool chain, while the ITBench run made no calls. Prompt format, task complexity, output budget, and serving cache differed, so the smoke result cannot predict the incident result. The partial expanded Param2 run covered only direct tasks and adds no evidence about tool use.

## Verification and limits

Seven local unit tests passed with `python3 -m pytest -q tests`. They cover synthetic scoring, the Param2 tool tag parser, the expanded task sequence scorer, final turn tool removal, and the exact entity proxy. Raw traces were inspected for actual tool calls and final answers. No official SRE judge was run. No model from the shared agentic SFT checkpoint was evaluated. The ten scenarios are a fixed subset of the 35 SRE scenarios, so the result is a focused failure analysis rather than a full benchmark estimate.

The next useful experiment is an inference and prompt ablation on the same ten snapshots: verify the model's native tool calling format and compare cache settings, then require a valid tool observation before accepting any diagnosis. Keep the existing raw runs unchanged as the baseline. A later official judge run would require a separately configured judge model.

## Reproduction

```sh
cd ~/professional-agent-evals/agent-eval-smoke-test
bash scripts/run_param2.sh
bash scripts/run_smoke.sh expanded
bash scripts/run_itbench_lite.sh param2 10 2048
bash scripts/run_itbench_lite.sh granite 10 2048 1
```

See [PARAM2_REPORT.md](PARAM2_REPORT.md) for the eight task breakdown and [ITBENCH_LITE_REPORT.md](ITBENCH_LITE_REPORT.md) for the focused ITBench comparison. The raw JSONL files and aggregate metrics use the identifiers in the run inventory above.
