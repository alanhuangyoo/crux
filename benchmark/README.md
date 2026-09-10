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

The board this targets is **Terminal-Bench 4.0** — 66 tasks, a flat 8h agent
budget, scored at `-k 5`, led by Opus 5 with Claude Code at **51.8%**. It is
neither of the two sets most of the numbers below come from, and worth
separating up front:

| | what it is | our result |
|---|---|---|
| **terminal-bench 4.0** | the board at tbench.ai, 66 tasks, 8h | in progress |
| terminal-bench 2.1 | the older 89-task set, ~900s median | 63.5% (k=3) |
| terminal-bench-pro | **Alibaba's separate benchmark**, 400 tasks | 65.7% (140/200, k=1) |

terminal-bench-pro is not a version of Terminal-Bench. It has its own publisher,
its own board, and its own leaders (Qwen3.5 + IFlow CLI at 51.5%). Scores across
the two are not comparable, and reading one board's numbers against the other's
run is a mistake this project has already made once.

### Terminal-Bench 2.1, 89 tasks

Self-hosted Qwen3.8-27B FP8 on four H20s. Six full runs, the last four at three
attempts per task:

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

### On 4.0 the failure changed shape

The wall-clock pressure that produced every seesaw above is gone: 2.1 gave a
~900s median and 27.8% of failures were timeouts; Pro gives 3600s and the median
task used 10% of it; 4.0 gives 8h. What is left is accuracy, and 4.0 pays only
for a clean sweep of a task's own suite.

That makes the task score a poor instrument at small samples and the test counts
a good one. Across the first 14 tasks of a k=1 run: **0 solved, 195 of 233
individual tests passed (83.7%), and 6 of 11 scored tasks were one or two tests
short of a sweep** — one of them 94 of 97. The agent's own checklist is the gap
it does not see: 35 checks bound against 206 graded tests, five against a
97-test suite.

The environment is clean here too — the reference solutions score **5 of 5**,
188 tests, including the four tasks whose suites run each test in a
privilege-dropped child.

- [x] Harbor installed, Docker verified, oracle re-checked on 2.1 (22/24) and 4.0 (5/5)
- [x] Crux v2 on Terminus 2 — live tmux session instead of one command per turn
- [x] Six full runs with a failure taxonomy and a measured variance floor
- [x] Quantisation, chat template, sampling params and effort default ruled out
- [x] Stock Terminus baseline — 56.9% against crux's 63.5%, paired z=+2.34
- [x] `crux solve` runs the benchmarked agent locally, not mini-swe-agent
- [ ] Claude Code and pi on the same model — is the gap the scaffold or the model?
- [ ] Full run at `-k 5`, `--upload`, submit

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
src/crux/local_agent.py  the same agent on this machine, with an approval gate
src/crux/local_env.py  a local shell shaped like the environment Terminus wants
src/crux/approval.py   what to ask before running; three levels, pi's no-UI rule
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
cp .env.example .env              # model endpoint and key
./scripts/smoke.sh                # verify the harness works here
```

Use it interactively -- multi-turn, resumable:

```bash
npm install -g @earendil-works/pi-coding-agent   # the terminal UI
crux chat                    # conversation, with the measured prompt sections
crux chat -c                 # continue the last session
crux chat --sections ""      # stock pi, the control arm
```

`crux chat` is a wrapper, and says so: pi supplies the front-end, crux supplies
the prompt sections and the model configuration. The alternative was writing a
REPL onto the Terminus base, whose `run()` builds a fresh chat every call, so
multi-turn would have meant monkeypatching chat construction. The same sections
run on the benchmark through `crux bench -a crux.pi_agent:CruxPiAgent`, so what
is measured is what runs.

Or run one task to completion, with the agent the benchmark measures:

```bash
crux solve "make the failing test pass" --cwd ~/work/project
```

It drives a live tmux session on this machine and asks before anything
destructive or outward-facing; `--approval never` is the benchmark's setting and
skips the asking, `--approval always` gates every command. With no tty, anything
that would need a prompt is refused rather than assumed — a run with nobody
watching should not delete something because there was no way to ask.

Run the benchmark:

```bash
crux bench                         # terminal-bench 4.0, the board's dataset
crux bench --dataset terminal-bench-pro/terminal-bench-pro
crux report <job-dir> [<other>]    # scores, failure taxonomy, or a diff
scripts/paired.py <base> <cand>    # compare two runs task by task, not by mean
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
