<h1 align="center">Crux</h1>

<p align="center">
  <b>A coding agent tuned against Terminal-Bench, and the measurement rig that decides what goes into it.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Terminal--Bench_2.1-0.773_pass@1-2ea44f" alt="pass@1 0.773">
  <img src="https://img.shields.io/badge/tests-4%2C388_passing-2ea44f" alt="4388 tests">
  <img src="https://img.shields.io/badge/base-pi_(Earendil_Works)-blue" alt="forked from pi">
  <img src="https://img.shields.io/badge/model-self--hosted_Qwen3.8--27B-8a2be2" alt="self-hosted model">
</p>

<p align="center">
  English · <a href="README.zh-CN.md">简体中文</a>
</p>

---

Crux is a fork of [pi](https://pi.dev) with changes to the agent loop and its
tools, together with the harness used to measure them on
[Terminal-Bench](https://www.tbench.ai/) 2.1.

The two live in one repository on purpose: every change below is attached to the
measurement that motivated it, and the arm that produced each number is kept
next to it. `packages/` is pi's code with the changes applied; `benchmark/` is
the harness. The CLI installs as both `crux` and `pi`.

## Results

Self-hosted Qwen3.8-27B, the 89 Terminal-Bench 2.1 tasks that run without a GPU
inside the container, 8× agent budget, pass@1 over whole runs.

| Configuration | pass@1 |
|---|---:|
| **crux** — 262,144-token window | **0.773** &nbsp;<sub>(0.793 on a second run)</sub> |
| crux — same code, 32,768-token window | 0.678 |
| crux — before the compaction and liveness fixes | 0.591 |
| Claude Code, same model | 0.730 |
| pi, unmodified | 0.539 |

The Claude Code row was measured in an earlier round at the 32,768 window and
has not been re-run since, so the comparison with the top row mixes a scaffold
difference with a deployment one.

Paired against the 32K run on the 86 tasks both scored: **12–4** on the sixteen
they disagree about, sign test z = +2.00.

Two runs of the winning configuration scored 0.773 and 0.793 and disagreed on 13
of 86 tasks, 7–6 — which is the noise floor this benchmark has at 89 tasks, and
the reason every claim here is a paired comparison rather than a difference of
totals.

<details>
<summary><b>Where the gain came from</b></summary>

<br>

Roughly half of it was making broken things work, and half was two deployment
settings that had been read as properties of the machine:

| | |
|---|---|
| 0.591 → 0.678 | compaction that could not fit its own request, and trials that spent 93% of their wall clock hung |
| 0.678 → 0.773 | an endpoint served at `--context-length 32768` against weights declaring 262,144, and eight concurrent trials that left the engine batching one request at a time |

The second also cut a full run from six hours to two.

</details>

## What changed

Each change is attached to the measurement that motivated it. None was reasoned
out in the abstract.

Rows marked † came after the last full run: each was checked against that run's
356 trajectories and has a test that fails without it, but none has been
measured end to end yet. `benchmark/docs/reference-comparison.md` has the
reference agent each one came from, and the designs the same trajectories ruled
out.

### Context and compaction

| Change | The measurement |
|---|---|
| A summarization request that fits the window it lives in | Compaction sends history as input and asks for the summary as output, both from one window, and nothing checked their sum: **2,115 of 2,370 compactions failed**, each appending its error and removing nothing |
| A rejected request halves and goes again | Characters per token is a property of the text — 3.99 on prose, 1.95 on compressed output — so no constant fits; a rejection is information |
| A summary cut off at the cap is kept | Discarding it is right on a 200K window and disables compaction on a 32K one, where it is the ordinary outcome |
| A keep budget spent by a trailing tool result still cuts | A cut point is never a tool result, so one large trailing result left the search with nowhere to cut and the compaction kept everything |
| A task run as one turn gets the full summary † | One task is one conversational turn, so every cut split it and took the short turn-prefix path: 4 of 9 compactions kept **2,302–4,443 characters of ~250,000 tokens**, against 7,113–13,193 through the full prompt |

### Liveness

| Change | The measurement |
|---|---|
| A connected stream that stops sending is failed | The SDK timeout covers getting a response, not keeping one: **25 timed-out trials had been silent for a median 112 of their 121 minutes** |
| A command with no timeout still has one | Ten minutes, since these tasks build things; a `grep -rl ... /` otherwise runs until the trial ends |
| A backgrounded process cannot hold a finished command open | The post-exit grace re-armed on every chunk, so a detached descendant writing to the inherited pipe re-armed it forever — one trial held its container for **eight hours after 48 seconds of work** |

### The loop

| Change | The measurement |
|---|---|
| A turn that answers nothing is not the agent finishing | `regex-chess` spent its whole output budget on thinking, stopped on `length`, and the run was recorded as settled having taken no action |
| Two-phase recovery for a truncated turn | Raise the ceiling and retry silently first; tell the model only if it truncates again |
| A turn cut off while thinking gets its reasoning back † | Reasoning is not replayed, so "pick up mid-thought" pointed at nothing: **292 turns in 60 trials** ended thinking at the limit, 43% followed by another that re-derived the same plan |
| A turn that stops inside its reasoning is a stall, not an answer † | 28 such turns in 14 trials; three runs ended on one, `gpt2-codegolf` at minutes 1, 2 and 24 with its deliverable never written |
| A tool call cut at the limit earns the raised ceiling too † | **All 42 cut tool calls** stopped at the initial 16K with the model's own 32K unused; 15 took two or more turns to get through |
| A deadline the loop can see | Six of twenty-four failed trials ended cut off with work in flight, while the median solve used a fraction of its budget |
| Stopping is questioned while the budget is unspent | Failed trials had spent a median 24% of their budget when they stopped; Claude Code's had spent 95% |
| A truncated round cannot buy every budget notice | Treating truncation as "earned another notice" removed the only brake for the case that repeats, and a truncating run collected all forty |
| A killed run is resumed, not scored zero | A cgroup OOM kill takes every process in the group, so a task that exhausts memory ends the run |

### Tools

| Change | The measurement |
|---|---|
| A default ceiling on one response | Claude Code's p99 output is 4,911 tokens against an 8K cap; the gap is what matters, not the cap |
| A failed edit shows the file, not a rule | "The old text must match exactly" left nowhere to go; edits failed 43 times in 528 calls against 156 reads |
| A truncated command keeps its first lines as well as its last † | A tail drops the first compiler error; 20 of 71 truncations were followed by the model going back for the part it lost |

### The prompt

| Change | The measurement |
|---|---|
| The model is told its reasoning is not kept † | Beside a tool call the visible text had a **median length of 0 characters**; the reasoning, dropped from every later request, a median of 1,016–1,349 |

## Method

Arms are compared task by task with a sign test rather than total against total,
and only when they ran concurrently — one p=0.041 became p=0.453 when the same
two arms were re-run in the same window. At the measured 15.5% flip rate, 89
tasks need an 11-point swing to reach p&lt;0.05 on their own, so an 8-point effect
needs replication.

Before a mechanism is tuned, it is counted. Compaction had its reserve and
keep-depth settings tuned for days before anyone asked how often it succeeded;
the answer was 14%.

A zero is only the agent's when the verifier ran. 14 of the last four arms'
zeros were the verifier's own test runner failing to install — two tasks can
never score in this environment, and one arm read as a regression at p = 0.057
through a GitHub outage, p = 0.73 once those were excluded. Each trial's time
budget follows harbor's limit for that task, which runs from 80 to 1,600
minutes, rather than one number for all of them.

`benchmark/docs/` keeps the candidate explanations that were measured and
rejected, which is most of them — thirty-four so far, several of them designs
that had already been written.

## Layout

```
packages/          pi, forked — agent core, model layer, TUI, coding agent CLI
benchmark/         the harness
  src/crux/        the harbor agent, prompt sections, endpoint tooling
  scripts/         preflight, paired comparison, mechanism health checks
  docs/            what was measured, including what it ruled out
  tests/           482 tests
```

## Running the benchmark

```bash
cd benchmark && uv sync
harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/<model>
```

`benchmark/scripts/preflight.sh` checks the endpoint, the launcher and the
harness checksum before a run starts. `benchmark/docs/ENVIRONMENT.md` covers the
evaluation host.

## Upstream

pi is by [Earendil Works](https://pi.dev) and is what makes this possible; see
[`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md). Upstream
issues and PRs belong at
[earendil-works/pi](https://github.com/earendil-works/pi), not here.

| Package | Description |
|---------|-------------|
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive coding agent CLI (installs as `crux` and `pi`) |
| **[@earendil-works/pi-agent-core](packages/agent)** | Agent runtime with tool calling and state management |
| **[@earendil-works/pi-ai](packages/ai)** | Unified multi-provider LLM API |
| **[@earendil-works/pi-tui](packages/tui)** | Terminal UI components |
| **[@earendil-works/chord](packages/chord)** | Application-composition runtime for services, RPC and plugins |

Licensed as upstream; see [`LICENSE`](LICENSE).
