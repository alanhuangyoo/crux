<h1 align="center">Crux</h1>

<p align="center">
  <b>A coding agent built on pi, re-engineered for long-horizon terminal tasks.<br/>Terminal-Bench 2.1 pass@1 from 0.539 to 0.773 — on par with Claude Code on the same model.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Terminal--Bench_2.1-0.773_pass@1-2ea44f" alt="pass@1 0.773">
  <img src="https://img.shields.io/badge/pi_%E2%86%92_Crux-0.539_%E2%86%92_0.773-2ea44f" alt="pi 0.539 to Crux 0.773">
  <img src="https://img.shields.io/badge/Claude_Code%2C_same_model-0.730-555555" alt="Claude Code on the same model 0.730">
  <img src="https://img.shields.io/badge/base-pi-blue" alt="built on pi">
</p>

<p align="center">
  English · <a href="README.zh-CN.md">简体中文</a>
</p>

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/results-dark.svg">
    <img src=".github/assets/results-light.svg" alt="Terminal-Bench 2.1 pass@1: Crux 0.773, Claude Code on the same model 0.730, pi upstream 0.539" width="760">
  </picture>
</p>

| Agent | Terminal-Bench 2.1 pass@1 |
|---|---:|
| **Crux** | **0.773** <sub>0.793 on a rerun</sub> |
| Claude Code, same model | 0.730 <sub>32K window</sub> |
| pi, upstream | 0.539 |

Self-hosted Qwen3.8-27B, identical weights for every agent · 89 tasks · pass@1
over whole runs.

## What changed from pi

All changes are in the agent layer; the model is untouched.

- **Context engine.** Compaction requests are sized to fit the window, retried
  smaller on rejection, and summarize single-task sessions in full; the task's
  original wording is carried verbatim through every compaction. Compaction
  success: **14% → 100%**.
- **Agent loop.** Recovery paths live in one explicit state machine: two-phase
  recovery from truncated output, the reasoning tail handed back after a cut,
  stall detection, and pacing against each task's own deadline. Truncated
  outputs: **18.0% → 0.7%**.
- **Runtime.** A stream idle watchdog, a default command timeout, process-tree
  reaping and resume after an OOM kill keep multi-hour runs alive. Idle time in
  timed-out runs: **94% → 3%**.
- **Tools.** A failed edit shows the nearest matching region of the file, long
  command output keeps both its head and its tail, and the model keeps short
  visible notes beside its tool calls.

## Evaluation

[`benchmark/`](benchmark) holds the harness behind these numbers: it runs the
agent in task containers via harbor, classifies every trial's outcome, and
compares versions task by task with a paired sign test.

```bash
cd benchmark && uv sync
harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/<model>
```

## Acknowledgements

Built on [pi](https://pi.dev) by Earendil Works
([earendil-works/pi](https://github.com/earendil-works/pi)). Licensed as
upstream; see [`LICENSE`](LICENSE).
