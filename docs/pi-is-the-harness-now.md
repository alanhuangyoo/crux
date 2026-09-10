# Porting the scaffold to pi, and what the trajectories said to port

crux started on Terminus. It runs on pi now, and the move exposed how much of
the score had been living in the wrapper rather than in the model.

Everything below is one deployment: Qwen3.8-27B-FP8 on four H20s, Terminal-Bench
2.1 (89 tasks, four GPU tasks excluded), 8x agent budget, arms run concurrently
so throughput is shared.

## Where the score actually sits

| Agent | Wrapper | pass@1 |
|---|---:|---:|
| Claude Code | — | 0.730 |
| crux-terminus | 1,290 lines | 0.719 |
| crux-pi, best | 386 lines | 0.640 |
| crux-pi, current arm of record | 386 lines | 0.588 |
| pi, stock | — | 0.539 |

Same model, same tasks, same budget. The 18-point spread between stock pi and
Claude Code is not the model answering wrong -- Claude Code solves those tasks
with the same weights behind it. It is scaffold, and the middle row is the
proof: a wrapper I wrote scores 0.719 with nothing clever in it, only a lot of
small interventions that pi's wrapper does not have.

## The seven changes, replicated

Two arms ran concurrently against one control. The changes are in pi itself --
loop, tools, settings -- not in a crux-side patch.

| Arm | pass@1 | pass@2 | paired vs control |
|---|---:|---:|---|
| control (stock pi, no ported prompt) | 0.500 | 52/89 | — |
| changed | **0.588** | **61/89** | 23-9, sign test p=0.020 |
| changed + `thinking=medium` | 0.584 | 52/89 | 19-7, p=0.029 |

The first round of this comparison read 8-2, p=0.109, and I could not tell
whether that was a real 8-point effect or noise -- 89 tasks at a 15.5% flip rate
need an 11-point swing to reach p<0.05, so the design could not resolve its own
result. Two concurrent replications settled it.

`thinking` is a harbor kwarg that had never been set on the pi side while Claude
Code's arms were pinned to `medium` the whole time. Aligning it changes nothing
here (12-12 against the other arm, p=1.0). It was still worth an arm: the
asymmetry was real and unexamined, and ruling it out is what leaves the +8.8
attributable to the loop and the prompt.

`filter-js-from-html` was solved for the first time by any scaffold, moving the
every-scaffold ceiling from 80/89 to 81/89.

## The largest difference measured anywhere in this project

For each trial, the share of its own budget spent at the moment the run ended:

| | solved | failed | failures under half |
|---|---:|---:|---:|
| crux-pi | 6% | **24%** | **47 of 73** |
| crux-terminus | 33% | 82% | 8 of 25 |
| Claude Code | 19% | **95%** | 7 of 24 |

When Claude Code fails, it has spent its run. When pi fails, it has spent a
quarter of one. Two thirds of pi's failures are voluntary stops: the loop reads
"no tool calls" as done, and the agent's word is a check it wrote for itself and
then passed.

This is not an effect at the edge of what 89 tasks can resolve. It is 3.4x, in
a directly observed quantity, over 73 failures.

crux-terminus already acts on this -- `_get_completion_confirmation_message`
adds the run's own spend to the "are you sure" that Terminus asks anyway, and
its failures now end at 82% of budget instead of pi's 24%. That method is 1,290
lines of scaffold away from pi. It is now in pi's loop instead, as
`stopBudgetShare`, off unless a harness sets `PI_STOP_BUDGET_SHARE`.

Smoke on three tasks, 30-minute budget: notices took openssl-selfsigned-cert
from 23 steps to 52 (scored), mteb-retrieve from 11 to 24, sanitize-git-repo
from 21 to 36. They also showed the first cut was wrong -- a fixed count of 8
notices was the binding constraint and the budget share never came near, so the
stop is productivity now: a notice whose round runs no tool call has found an
agent with nothing left to do.

## What is left, and what it costs

The best pi arm splits its 89 tasks:

| | | worth |
|---|---:|---|
| solid (2/2) | 43 | — |
| flaky (1/2) | 18 | **+10 points, no new capability** |
| zero (0/2) | 28 | the rest |

Ten of the twenty-one points to 80 need nothing solved that has not already been
solved once. Of the 28 zeros, 25 have been solved by some arm on this box, and
crux-terminus solved most of them -- so the shortlist for porting is not a
research question, it is a diff between two wrappers I already have.

Only `make-doom-for-mips`, `make-mips-interpreter` and `raman-fitting` have never
been solved by anything, including the oracle arms.

Reading the 18 flaky tasks for a shared failure shape found none: the failing
attempt is not systematically longer or shorter than the passing one (10-8,
p≈0.8). That was the fourteenth pattern this project has proposed and measured
away.
