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

Terminal-Bench 2.1, 89 tasks (the 4 GPU-only ones excluded), self-hosted
Qwen3.8-27B FP8 on four H20s. Six full runs, the last four at three attempts
per task:

| arm | trials | solved | timeout | wrong | tokens/trial |
|---|---|---|---|---|---|
| concurrency 24 | 89 | 51.7% | 33% | — | — |
| concurrency 8 | 87 | 62.1% | 27.6% | 10.3% | 54,297 |
| + output cap 16384 | 87 | 52.9% | 33.3% | 13.8% | 42,250 |
| xhigh effort | 263 | **63.5%** | 27.8% | 8.7% | 51,153 |
| medium effort | 265 | 62.3% | **19.2%** | 18.5% | **23,351** |
| BF16 weights | 255 | 61.6% | 29.4% | 9.0% | 54,636 |
| medium + verify | 265 | 62.6% | 23.3% | 13.9% | 24,740 |

**The score stopped separating the arms; the failure split did not.** The last
four sit between 61.6% and 63.5%, all inside a variance floor measured at ±2–3
points — a fifth of the tasks change outcome between runs of the *same*
configuration. Read as means they are one result. Read as failure splits they
are four different things, and only one bucket is reachable from this repo:
timeouts are capped by 129 tok/s on this hardware, wrong answers are not.

Two findings carried the work. Concurrency turned out to be a scoring setting
rather than a throughput one — the per-task timeout is wall clock, so every
extra concurrent trial takes tokens per second from all the others — and
lowering it from 24 to 8 was worth 9 points *and* two hours, because a
timed-out task holds its slot for the full budget while a solved one gives it
back early. And `reasoning_effort`, which every earlier measurement had been
taking at the model's most expensive default without saying so, is a seesaw:
halving generation cuts timeouts by 8 points and adds 8 points of wrong
answers, moving failures from a bucket bounded by hardware into one bounded by
the prompt.

### The environment is not the gap

Checked directly, because it is the first thing to suspect: the dataset is the
official package pinned by sha256, the runner is harbor 0.22.0, grading is each
task's own tests (absent from the container during the agent phase), and
per-task timeouts carry no override or multiplier. Running the reference
solutions scores **22 of 24**. Both failures are tasks that rotted —
`build-pov-ray` fetches source from a 1993 archive that now answers 403, and
`mcmc-sampling-stan` pins StanHeaders but lets RcppParallel float to a version
that requires a cmake its image does not ship. Neither is reachable from here,
and together they are worth about two points.

What is left is hardware. Of the failures, generation alone consumes 55–110% of
the task budget on the timeout-bound ones, against a measured ceiling of 129
tok/s for TP=4 on an H20 — speculative decoding lost in five configurations,
TP=8 was worse, torch.compile ran out of memory. Terminal-Bench times by the
wall clock and an H20 has roughly 15% of an H100's compute.

- [x] Harbor installed, Docker verified, oracle re-checked on TB 2.1
- [x] Crux v2 on Terminus 2 — live tmux session instead of one command per turn
- [x] Six full runs with a failure taxonomy and a measured variance floor
- [x] Quantisation, chat template, sampling params and effort default ruled out
- [ ] Stock Terminus baseline — the scaffold the model card's 73.0 used
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
