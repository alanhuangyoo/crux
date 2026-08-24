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

Phase 0 complete — harness verified on the dev box (oracle 2/2, mean 1.000).
Evaluation runs on the x86-64 dev machine, not this Mac; see docs/ENVIRONMENT.md.

- [x] Harbor installed, Docker verified, oracle smoke test green on dev
- [ ] Phase 1 — baseline: existing agent + DeepSeek, small subset
- [x] Phase 2 — Crux v1: model client + ATIF trajectory emission
- [ ] Phase 3 — iterate on prompts / tools / context management
- [ ] Phase 4 — full run (all tasks x >=5 trials), `--upload`, submit PR

## Layout

```
src/crux/agent.py   the agent — a bash loop, Harbor BaseAgent
scripts/smoke.sh        oracle run; verifies the harness, not the agent
scripts/baseline.sh     Phase 1 baseline against an existing agent
docs/RESEARCH.md        competitive landscape, submission rules, red lines
```

## Tests

The ATIF trajectory is the artifact Harbor's CI validates a submission
against, so it is covered by tests that run Harbor's own validator:

```bash
HARBOR_SP=$(dirname $(dirname $(readlink -f $(which harbor))))/lib/python3.14/site-packages
PYTHONPATH="src:$HARBOR_SP" .venv/bin/python -m pytest tests/ -q
```

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
