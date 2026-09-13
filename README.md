# Crux

A coding agent, and the measurement rig that decides what goes into it.

Crux is a fork of [pi](https://pi.dev). Everything in `packages/` is pi's, plus
a set of changes to the agent loop and its tools; everything in `benchmark/` is
the harness that measures those changes against
[Terminal-Bench](https://www.tbench.ai/) 2.1. The two live together because
that is the only arrangement in which a claim about the agent can be checked:
the diff against upstream is the work, and the arm that produced each number is
recorded next to it.

The CLI installs as both `crux` and `pi`, and is the same binary as upstream's
with the changes below applied.

## Where it stands

One self-hosted model, the 89 Terminal-Bench 2.1 tasks that run without a GPU
inside the container, 8x agent budget, pass@1 over whole runs.

**Qwen3.8-Flash-Next-FP8, served at the 262,144-token window the weights
declare:**

| | pass@1 |
|---|---:|
| **crux** | **0.773** |
| crux, same code at a 32,768 window | 0.678 |
| crux, before the compaction and hang fixes | 0.591 |

Paired against the 32K run on the 86 tasks both scored: 12-4 on the sixteen they
disagree about, sign test z=+2.00.

**Qwen3.8-27B, 32,768 window** -- the earlier regime, kept because the
cross-scaffold comparison was made there and has not been repeated:

| | pass@1 |
|---|---:|
| Claude Code, same model | 0.730 |
| crux-terminus, this project's earlier Terminus-based scaffold | 0.719 |
| crux | 0.588 |
| pi, unmodified | 0.539 |

The two tables are not comparable: different weights, different window,
different engine load. Numbers measured in one regime are not carried into the
other, which is why the old one is still here rather than overwritten.

## What is changed, and why each change is there

Each one is taken from a shipped reference implementation rather than reasoned
out, and each is attached to the measurement that motivated it.

| Change | Taken from | The measurement |
|---|---|---|
| A turn that answers nothing is not the agent finishing | — | `regex-chess` spent 65,536 output tokens entirely on thinking, `stopReason: "length"`, and the run was recorded as settled having taken no action |
| Two-phase recovery for a truncated turn | Claude Code's loop | Raise the ceiling and retry silently first; tell the model only if it truncates again |
| A default ceiling on one response | Claude Code's token budget | Their p99 output is 4,911 tokens against an 8K cap; the quantity that matters is the gap, not the cap |
| A deadline the loop can see | — | Six of twenty-four failed trials ended cut off with work in flight, while the median solve used a fraction of its budget |
| A bash command the model did not bound is still bounded | Claude Code's tiered shell policy | 85.7% of 6,100 bash calls carried no timeout; 13 of 401 trials ended on a tool call that never finished |
| Stopping is questioned while the budget is unspent | Terminus's completion confirmation | pi's failed trials had spent a median 24% of their budget when they stopped; Claude Code's had spent 95% |
| A killed run is resumed, not scored zero | — | A cgroup OOM kill takes every process in the group, so a task that exhausts memory ends pi's run and merely ends Terminus's command |
| A summarization request that fits the window it lives in | — | Compaction sends the history as input and asks for the summary as output, both out of one window, and nothing checked their sum: 2,115 of 2,370 compactions failed, each appending its error and removing nothing, so one trial compacted 91 times across 74 turns and climbed 106 tokens per attempt |
| A rejected request halves and goes again | — | Characters per token is a property of the text, measured from 3.99 on prose to 1.95 on what `write-compressor` accumulates, so no constant fits; a rejection is information |
| A summary cut off at the cap is kept | — | Discarding it is right on a 200K window and disables compaction on a 32K one, where it is the ordinary outcome |
| A connected stream that stops sending is failed | — | The SDK's timeout covers getting a response, not keeping one: 25 timed-out trials had been silent for a median 112 of their 121 minutes, and `install-windows-3.11` held its container for eight hours after 48 seconds of work |
| A command with no timeout still has one | Claude Code's default | Ten minutes rather than their two, because these tasks build things; a `grep -rl ... /` over the whole filesystem otherwise takes the trial with it |
| A backgrounded process cannot hold a finished command open | — | The post-exit grace re-armed on every chunk, so a detached descendant writing to the inherited pipe re-armed it forever |

The first seven, measured on Qwen3.8-27B against a matched control on the same
tasks in the same window: **0.588 against 0.500, 23-9 on paired tasks, sign test
p=0.020**, replicated by a second concurrent arm at 0.584. The last six were
measured on Flash-Next and are the 0.591 to 0.678 step in the table above; the
window that follows it is not theirs.

## Layout

```
packages/          pi, forked -- agent core, model layer, TUI, coding agent CLI
benchmark/         the rig that measures it
  src/crux/        the harbor agent, prompt sections, endpoint tooling
  docs/            what was measured, and what it ruled out
  tests/           464 tests, including one per harness bug found the hard way
```

## Running the benchmark

```bash
cd benchmark && uv sync
harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/<model>
```

`benchmark/docs/ENVIRONMENT.md` covers the evaluation host, and
`benchmark/docs/pi-is-the-harness-now.md` covers why the agent work sits in this
repository rather than beside it.

## On method

Arms are compared task by task with a sign test, never total against total, and
only when they ran concurrently -- one p=0.041 collapsed to p=0.453 when the
same two arms were re-run in the same window. 89 tasks at the measured 15.5%
flip rate need an 11-point swing to reach p<0.05 on their own, so an 8-point
effect is established by replication rather than by one reading.

Two of the largest gains were not code. The endpoint was served with
`--context-length 32768` against weights declaring 262,144, and the KV pool held
4.6M tokens all along; and eight concurrent trials left the engine running one
to two requests at a time, because a trial spends most of its life in bash
rather than waiting on the model. Both were numbers that were correct where they
were chosen and were then read as properties of the box. The first cost a day of
fixing its consequences, the second four hours per iteration.

Twenty-two candidate explanations for the gap have been measured and rejected,
several of them designs that were already written. `benchmark/docs/` keeps them,
because the rejected ones are most of the information.

## Upstream

pi is by [Earendil Works](https://pi.dev) and is what makes this project
possible; see [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md) and
the package list below. Upstream issues and PRs belong at
[earendil-works/pi](https://github.com/earendil-works/pi), not here.

| Package | Description |
|---------|-------------|
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive coding agent CLI (installs as `crux` and `pi`) |
| **[@earendil-works/pi-agent-core](packages/agent)** | Agent runtime with tool calling and state management |
| **[@earendil-works/pi-ai](packages/ai)** | Unified multi-provider LLM API |
| **[@earendil-works/pi-tui](packages/tui)** | Terminal UI components |
| **[@earendil-works/chord](packages/chord)** | Application-composition runtime for services, RPC and plugins |

Licensed as upstream; see [`LICENSE`](LICENSE).
