# Crux

A terminal agent scaffold targeting [Terminal-Bench](https://www.tbench.ai/),
built around a specific bet: **the official leaderboard has no current-generation
DeepSeek entry, and the gap between what it records and what the model can
actually do is enormous.**

| Leaderboard | Entries | DeepSeek entries |
|---|---|---|
| terminal-bench 2.0 | 142 | **1** — Terminus 2 + DeepSeek-V3.2, **39.6%** (rank 86, 2026-02-10) |
| terminal-bench 2.1 | 17 | **0** |

Third-party evaluations put current DeepSeek models around **82–88%** on the
same tasks. Nobody has submitted that to the official board.

## Goal

Build a scaffold that lands a current-generation DeepSeek model on the official
Terminal-Bench leaderboard, and do it at a fraction of the cost of the
frontier-lab submissions above it ($134–$2,059 per evaluation today).

This is a scaffold project, not a model project. The reference point is that
the same model scores very differently under different scaffolds — Terminal-Bench's
own data shows a **17%** swing for one model between two harnesses.

## Status

Full runs are working. Current best on Terminal-Bench 2.1 (89 tasks, the 4
GPU-only ones excluded), against a self-hosted Qwen3.8-27B FP8 on four H20s:

| | concurrency | score | solved | timeouts | wall clock |
|---|---|---|---|---|---|
| baseline | 24 | 51.69% | 46/89 | 33% | 6h03m |
| **current** | **8** | **60.67%** | **54/89** | 28% | **3h56m** |

Concurrency is a scoring setting, not a throughput one: the benchmark's
per-task timeout is wall clock, so every extra concurrent trial takes tokens
per second away from all the others. Lowering it is worth 9 points *and* two
hours — a timed-out task holds its slot for the full budget by definition,
while a solved one gives the slot back early. See [docs/ABLATION.md](docs/ABLATION.md).

Of the 35 remaining failures, roughly 19 are bound by generation speed
(generation alone consumes most of the task budget) and 13 are not — wrong
answers, or trials that sat waiting on a command. That bounds what more speed
can buy.

- [x] Harbor installed, Docker verified, oracle smoke test green
- [x] Crux v1 on mini-swe-agent: model client + ATIF trajectory emission
- [x] Crux v2 on Terminus 2 — live tmux session instead of one command per turn
- [x] Full 89-task runs, with a failure taxonomy per run
- [ ] Close the gap to the model card's 73.0
- [ ] Full run at >=5 trials per task, `--upload`, submit PR

## What this is

Crux is a **modified scaffold, not a new agent** — the loop, parser,
trajectory export and format contract are upstream's. Writing that machinery
from scratch was tried first and reached 4.62%; the gap is the iterations
already baked into a mature agent, not prompt tuning.

There are two, and which base you sit on outweighs every prompt change measured
here:

- `crux-terminus` (`terminus_agent.py`) — subclasses Harbor's **Terminus 2**,
  which drives a live tmux session and sends keystrokes. This is what the runs
  above use. It adds Crux's all-or-nothing scoring instruction and `crux
  submit`, and nothing else.
- `crux` (`agent.py`) — the original, on **mini-swe-agent**: one shell command
  per turn, read its stdout. Kept for comparison. It cannot express entering an
  interactive session at all, which decides whole tasks (`qemu-alpine-ssh`).

Around it is the part that actually drives the work: a pipeline that turns a
run into an answer to *why did we score that*. Every change in this repo came
from a distinction the leaderboard number does not make.

## Layout

```
src/crux/terminus_agent.py  CruxTerminusAgent — subclasses Harbor's Terminus 2
src/crux/agent.py      CruxAgent — the earlier mini-swe-agent base
src/crux/prompts.py    the modified templates; upstream's format contract intact
src/crux/config.py     knobs and ablation variants (`stock` = upstream prompt)
src/crux/analysis.py   job dir -> scores, failure taxonomy, completion fractions
scripts/analyze.py     report on one run, or diff two
scripts/provision.sh   set up an eval node (every setting has a run behind it)
scripts/run.sh         run Crux; scripts/smoke.sh runs oracle to check the box
deploy/                engine, router and launch scripts for the eval box
docs/DEPLOYMENT.md     the four-card serving recipe, and what not to use
docs/ENVIRONMENT.md    what broke and why — btrfs, address pools, GPU tasks
docs/ABLATION.md       every measurement, including the ones that overturned
                       an earlier conclusion in this file
```

## Tests

The ATIF trajectory is the artifact Harbor's CI validates a submission
against, so it is covered by tests that run Harbor's own validator:

```bash
HARBOR_SP=$(ls -d ~/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
PYTHONPATH="src:$HARBOR_SP" uv run --with pytest python -m pytest tests/ -q
```

118 tests. Several are guards rather than unit tests — `stock` must render
upstream's prompt untouched, and the agent name must stay `crux-terminus`,
because a silently substituted agent has already invalidated a full run here.

## Quickstart

```bash
uv tool install 'harbor[modal]'   # provides `harbor`
cp .env.example .env              # fill in DEEPSEEK_API_KEY
./scripts/smoke.sh                # verify the harness works here
```

## Rules that shape the design

Harbor's CI rejects submissions automatically, so several constraints are
architectural rather than optional:

- **ATIF trajectories are mandatory** for every passing trial — an agent that
  only writes custom logs fails validation outright. Hence `SUPPORTS_ATIF = True`.
- **No timeout or resource overrides.** `timeout_multiplier` must be unset or `1.0`.
- **Every task, >= 5 trials each.** Errored trials count as reward 0 and may not
  be dropped.
- Successful trajectories are audited by an LLM judge for reward hacking.

Full details, including the enforcement history, in [docs/RESEARCH.md](docs/RESEARCH.md).
