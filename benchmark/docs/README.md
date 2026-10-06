# Engineering reports

The design record behind Crux. Every change to the agent was proposed, measured
against real trajectories, and either adopted or ruled out here — with the
numbers that decided it.

## Start here

| Report | What it covers |
|---|---|
| [Design review against the reference agents](reference-comparison.md) | Mechanisms from Claude Code, Codex, opencode, hermes-agent and grok-build, each checked against 356 Terminal-Bench trajectories: what was adopted, what the data ruled out, and a within-task analysis of what separates a solved run from a failed one |

## Analysis

| Report | What it covers |
|---|---|
| [What the failures are](what-the-failures-are.md) | Failure analysis across Terminal-Bench 2.1, SWE-bench Verified and SWE-Atlas-QnA: the attribution chain, where the agent's actions go, the measurement pitfalls of agent benchmarks and the guards built against them |
| [Porting the scaffold to pi](pi-is-the-harness-now.md) | Moving the agent onto pi, and the trajectory evidence that decided what to carry over |
| [Agent-loop and tool changes](../../docs/crux-findings.md) | Each change to pi's loop and tools, the reference implementation it came from, and the measurement behind it |
| [The branch, change by change](../../docs/crux-branch.md) | What the branch adds on top of upstream pi, and the question each change answers |
| [Ablations](ABLATION.md) *(中文)* | Context window, output ceiling, reasoning level, concurrency and FP8, run as paired arms; the statistics of slice sizes and full runs |

## Environment

| Report | What it covers |
|---|---|
| [Serving setup](DEPLOYMENT.md) *(中文)* | Qwen3.8-27B (FP8) on SGLang, tensor-parallel over four GPUs |
| [Evaluation host](ENVIRONMENT.md) *(中文)* | Where evaluations run and why |
| [Competitive landscape](RESEARCH.md) *(中文)* | Terminal-Bench versions, leaderboards and submissions, surveyed from primary sources |
