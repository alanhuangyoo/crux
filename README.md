# Crux

A fork of [pi](https://pi.dev) with changes to the agent loop and its tools,
plus the harness used to measure them on
[Terminal-Bench](https://www.tbench.ai/) 2.1.

`packages/` is pi's code with the changes applied. `benchmark/` is the harness.
The CLI installs as both `crux` and `pi`.

## Results

Self-hosted Qwen3.8-27B, the 89 Terminal-Bench 2.1 tasks that run without a GPU
inside the container, 8x agent budget, pass@1 over whole runs.

| | pass@1 |
|---|---:|
| crux, 262,144-token window | 0.773 |
| crux, 32,768-token window | 0.678 |
| crux, before the compaction and hang fixes | 0.591 |
| Claude Code | 0.730 |
| pi, unmodified | 0.539 |

The first three rows are the same code at different settings. The Claude Code
and unmodified-pi rows were measured in an earlier round at the 32,768 window
and have not been re-run since, so they are not directly comparable to the top
row.

## Changes

| Change | Why |
|---|---|
| A turn that answers nothing is not the agent finishing | `regex-chess` spent its whole output budget on thinking, stopped on `length`, and the run was recorded as settled having taken no action |
| Two-phase recovery for a truncated turn | Raise the ceiling and retry silently first; tell the model only if it truncates again |
| A default ceiling on one response | Claude Code's p99 output is 4,911 tokens against an 8K cap; the gap is what matters, not the cap |
| A deadline the loop can see | Six of twenty-four failed trials ended cut off with work in flight, while the median solve used a fraction of its budget |
| A bash command the model did not bound is still bounded | 85.7% of 6,100 bash calls carried no timeout; 13 of 401 trials ended on a tool call that never finished |
| Stopping is questioned while the budget is unspent | Failed trials had spent a median 24% of their budget when they stopped; Claude Code's had spent 95% |
| A killed run is resumed, not scored zero | A cgroup OOM kill takes every process in the group, so a task that exhausts memory ends the run |
| A summarization request that fits the window it lives in | Compaction sends history as input and asks for the summary as output, both from one window, and nothing checked their sum: 2,115 of 2,370 compactions failed, each appending its error and removing nothing |
| A rejected request halves and goes again | Characters per token is a property of the text — 3.99 on prose, 1.95 on compressed output — so no constant fits |
| A summary cut off at the cap is kept | Discarding it is right on a large window and disables compaction on a small one |
| A connected stream that stops sending is failed | The SDK timeout covers getting a response, not keeping one: 25 timed-out trials had been silent for a median 112 of their 121 minutes |
| A command with no timeout still has one | Ten minutes, since these tasks build things; a `grep -rl ... /` otherwise runs until the trial ends |
| A backgrounded process cannot hold a finished command open | The post-exit grace re-armed on every chunk, so a detached descendant writing to the inherited pipe re-armed it forever |

Two settings outside the code mattered as much. The endpoint had been served
with `--context-length 32768` against weights declaring 262,144, with a KV pool
that held 4.6M tokens; and eight concurrent trials left the engine running one
to two requests at a time, since a trial spends most of its life in bash rather
than waiting on the model. Raising both is the 0.678 to 0.773 step, and cut a
full run from six hours to two.

## Layout

```
packages/          pi, forked -- agent core, model layer, TUI, coding agent CLI
benchmark/         the harness
  src/crux/        the harbor agent, prompt sections, endpoint tooling
  docs/            what was measured, including what it ruled out
  tests/           464 tests
```

## Running the benchmark

```bash
cd benchmark && uv sync
harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/<model>
```

`benchmark/scripts/preflight.sh` checks the endpoint and the launcher before a
run. `benchmark/docs/ENVIRONMENT.md` covers the evaluation host.

## Method

Arms are compared task by task with a sign test rather than total against
total, and only when they ran concurrently — one p=0.041 became p=0.453 when the
same two arms were re-run in the same window. At the measured 15.5% flip rate,
89 tasks need an 11-point swing to reach p<0.05 on their own, so an 8-point
effect needs replication.

`benchmark/docs/` keeps the candidate explanations that were measured and
rejected, which is most of them.

## Upstream

pi is by [Earendil Works](https://pi.dev); see [`AGENTS.md`](AGENTS.md) and
[`CONTRIBUTING.md`](CONTRIBUTING.md). Upstream issues and PRs belong at
[earendil-works/pi](https://github.com/earendil-works/pi), not here.

| Package | Description |
|---------|-------------|
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive coding agent CLI (installs as `crux` and `pi`) |
| **[@earendil-works/pi-agent-core](packages/agent)** | Agent runtime with tool calling and state management |
| **[@earendil-works/pi-ai](packages/ai)** | Unified multi-provider LLM API |
| **[@earendil-works/pi-tui](packages/tui)** | Terminal UI components |
| **[@earendil-works/chord](packages/chord)** | Application-composition runtime for services, RPC and plugins |

Licensed as upstream; see [`LICENSE`](LICENSE).
