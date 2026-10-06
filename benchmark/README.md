# Crux evaluation system

The harness that runs Crux on [Terminal-Bench](https://www.tbench.ai/) and
decides which changes to the agent go in. The agent itself lives at the
repository root — see [`../README.md`](../README.md).

It does four things:

- **Runs the agent at scale.** A harbor agent
  installs Crux inside each task container, configures it for the model
  endpoint, and paces it to the task's own time limit.
- **Explains every score.** Each trial is classified automatically — solved,
  false completion, out of budget, timeout, context exhausted, format error, or
  an environment failure that never reached the agent.
- **Compares changes honestly.** Two arms are compared task by task with a sign
  test, restricted to the window where both were running, so run-to-run noise
  is not mistaken for an effect.
- **Gates every launch.** Preflight checks the endpoint, the launch settings and
  the harness checksum before an arm spends GPU hours.

## Layout

```
src/crux/
  pi_agent.py         the harbor agent: installs Crux in the container, declares the
                      model's context window and reasoning, per-task time budgets,
                      resume after an OOM kill
  prompts.py          prompt sections appended to the system prompt, chosen per arm
  analysis.py         failure taxonomy, completion fractions, paired comparison
  cli.py              the `crux-bench` command line
  terminus_agent.py   Terminus 2 baseline, on the same model
  agent.py            mini-swe-agent baseline, on the same model
  doctor.py, probe.py endpoint checks and quick configuration probes
  watch.py            live view of runs in flight

scripts/
  one.sh              one arm, for an absolute number
  ab.sh               two arms launched together, for a comparison
  targeted.sh         one arm over the tasks a change aims at, plus regression guards
  preflight.sh        checks the endpoint, launch settings and harness checksum
  analyze.py          report on a run, or diff two
  paired.py           task-by-task comparison with a sign test
  paired-window.py    the same, restricted to the window both arms shared
  compaction-health.py, hang-profile.py   mechanism health checks
  provision.sh        set up an evaluation node
  smoke.sh            end-to-end check of the harness with the oracle agent
  swe-bench-verified.sh, swe-atlas.sh     the same agent on SWE-bench and SWE-Atlas

tests/                486 tests
```

## Quick start

```bash
uv sync
cp .env.example .env                  # model endpoint and key
./scripts/smoke.sh                    # verify the harness on this host
./scripts/preflight.sh scripts/one.sh # check the launch before it runs
./scripts/one.sh <name> <bundle>      # one full arm
```

Read the results:

```bash
crux-bench report <job-dir>                 # score, failure taxonomy, near misses
crux-bench report <job-dir> <other-job-dir> # what changed, task by task
scripts/paired.py <baseline> <candidate>    # paired sign test
```

Run a task locally with the agent the benchmark measures:

```bash
crux-bench solve "make the failing test pass" --cwd ~/work/project
```

## Tests

```bash
uv run pytest -q
```

The suite pins the decisions the analysis depends on: environment failures are
never charged to the agent, a scoring harness's pytest wrapper is never read as
its score, every launcher carries the same settings, and preflight reads only
the launcher it is given.
