# ITBench Lite SRE comparison

Run date: October 1, 2026. Host: `passpoli`. Dataset: `ibm-research/ITBench-Lite`, revision `d0916b08ba421ce5e672e9ad68aa947d938dfef0`. Ten pinned SRE snapshots: scenarios 1, 2, 4, 5, 6, 7, 8, 9, 11, and 12. Scenario numbers 3 and 10 do not exist in this dataset revision.

Both models received the same incident prompt, six read only reference SRE tools, a 2,048 token generation cap, eight investigation turns, one final turn without tools, and a 600 second per incident time limit. Temperature was zero. Granite 4.2 3B ran through Ollama with reasoning disabled. Param2 17B A2.4B Thinking ran through Transformers with generation cache enabled. The Param2 checkpoint was pinned to revision `86fd8f7bb96a404aaab589cb56831f575bbb92b5`.

The score is a deterministic proxy: a diagnosis counts only if an entity marked `contributing_factor: true` matches a ground truth root cause entity by namespace, kind, and resource name pattern. It is not the official ITBench SRE judge score. The official evaluator needs a separate judge model, which was not configured. A structurally valid diagnosis here means a JSON object with the three required arrays; it does not imply factual validity.

| Model | Root cause entity hits | Structurally valid diagnoses | Tool calls | Median incident latency |
| --- | ---: | ---: | ---: | ---: |
| Param2 17B A2.4B Thinking | 0/10 | 10/10 | 0 | 158.2 s |
| Granite 4.2 3B | 0/10 | 2/10 | 80 | 17.1 s |

| Scenario | Param2 hit | Param2 calls | Granite hit | Granite calls | Granite final response |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 0 | 0 | 8 | Extra JSON data |
| 2 | 0 | 0 | 0 | 8 | Textual tool call |
| 4 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 5 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 6 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 7 | 0 | 0 | 0 | 8 | Wrong top level shape |
| 8 | 0 | 0 | 0 | 8 | Valid JSON, wrong entity |
| 9 | 0 | 0 | 0 | 8 | Extra JSON data |
| 11 | 0 | 0 | 0 | 8 | Textual tool call |
| 12 | 0 | 0 | 0 | 8 | Valid JSON, empty diagnosis |

## Failure analysis

Param2 skipped investigation entirely. It described a plan to call all six tools in its reasoning, then emitted a final JSON object without a tool call. All ten scenarios received the exact same 1,409 token answer, byte for byte. This follows from the identical prompt and zero snapshot observations, not from the scorer. The answer cited `alert_summary` and `event_analysis` despite neither tool running, invented a checkout pod and its resource spec, used names outside the requested `namespace/Kind/name` form, repeated one entity three times, and marked every entity `contributing_factor: false`. It therefore could not score a root cause hit even if the invented name happened to match. Its schema success is misleading: the content is unsupported by the incident data.

Granite did investigate. Every scenario used all eight available tool calls, but none identified the scored root cause. It often queried unsupported event columns, guessed metric patterns that matched no files, and spent more calls correcting the query. On the forced final turn, two outputs were textual tool calls even though tools were unavailable. Six others had extra JSON or a wrong top level shape. Only scenarios 8 and 12 met the minimal JSON schema; scenario 8 named a nonmatching entity, and scenario 12 returned empty arrays.

These results should not be compared to an official ITBench leaderboard. The run used six reference tools rather than the full agent environment and used exact entity matching instead of the official LLM judge. The serving configurations also differ: Granite reasoning was disabled while Param2 used its thinking checkpoint. The useful conclusion is narrower and direct: under this shared offline snapshot harness, Param2 failed to consume evidence and Granite failed to produce a reliable final diagnosis.

## Artifacts and reproduction

Raw trajectories are in `results/itbench-lite-param2-20261001T002109.jsonl` and `results/itbench-lite-granite-20261001T001649.jsonl`. They contain tool observations, raw responses, usage, and per scenario timing. The scripts and scorer are in `scripts/run_itbench_lite.sh` and `eval_smoke/itbench_lite.py`. Run the same ten scenarios on `passpoli` after both GPUs have capacity:

```sh
bash scripts/run_itbench_lite.sh param2 10 2048
bash scripts/run_itbench_lite.sh granite 10 2048 1
```

Local unit tests: `python3 -m pytest -q tests`, seven passed.
