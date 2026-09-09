# What was measured, and what it ruled out

Seven changes to pi's agent loop and tools, each taken from a shipped reference
implementation rather than reasoned out, and each with the measurement that
motivated it. Then seven explanations for the remaining gap, all of which the
data rejected.

The benchmark throughout is Terminal-Bench 2.1, 89 tasks, one self-hosted
Qwen3.8-27B behind an OpenAI-compatible endpoint, 8x agent budget, arms run
concurrently so throughput is shared.

## The changes

| | Taken from | The measurement that motivated it |
|---|---|---|
| A turn that answers nothing is not the agent finishing | — | `regex-chess` spent 65,536 output tokens, 199,246 characters, entirely thinking, `stopReason: "length"`, and the run was recorded as settled having taken no action |
| Two-phase recovery for a truncated turn | Claude Code `the-loop.mdx` (`max_output_tokens_escalate` then `_recovery`) | The order was the part I had backwards: raise the ceiling and retry silently first, tell the model only if it truncates again |
| A default ceiling on one response | Claude Code `token-budget.mdx` | Their p99 output is 4,911 tokens and they cap at 8K; over 10,225 turns here p99 is 16,161, so 16K truncates 0.4% where 8K truncates 3% |
| A deadline the loop can see | — | Six of pi's twenty-four failed trials ended cut off with work in flight; the median solve used a fraction of its budget |
| A bash command the model did not bound is still bounded | Claude Code's tiered shell policy | 85.7% of 6,100 bash calls carried no timeout; 13 of 401 trials ended on a tool call that started and never finished |
| Settings delivered where the run will read them | — | harbor builds the exec env from the model connection, so a variable exported on the runner reaches nothing inside the container |
| An offline bundle that works on Alpine | — | Three of eleven SWE-Atlas images are musl; a glibc node reports the mismatch as `env: can't execute 'node'` |

They demonstrably act: across one arm, `Output token limit hit` appears in 29
trials against 0 in the control, and `stopReason: "length"` 241 times against
96 — the ceiling biting and each turn being retried rather than ending the run.

**They do not demonstrably score.** 79 pairs, 8 gains, 2 losses, p=0.109, and
the gains do not concentrate where the change acted (4:1 in the 28 pairs it
touched, 4:1 in the 51 it did not). A replication is running to pool.

## What the gap is not

claude-code leads pi by 17 points on this model (72.7% against 55.7%, 88
pairs, p=0.006). Seven explanations were checked and rejected:

| Explanation | How it died |
|---|---|
| Sub-agents / swarm | 0 of 89 claude-code trials spawned one |
| The todo tools (7.8% of its calls) | Controlling for run length: 69.7% against 66.7% |
| MicroCompact context management | pi's peak context is 14% of the window at the median; 1 trial in 178 passed half |
| TOKEN_BUDGET nudges | They exist to make an agent stop sooner; pi already stops too soon |
| "It thinks more between actions" | A units error. Its assistant messages carry one block each, so 115 "turns" is 40 tool calls |
| "pi writes over-precise regexes" | Backwards: length-bounded patterns are 3.8% of claude-code's greps, 1.3% of pi's |
| Prefix caching | pi 96.4%, claude-code 27.2% |
| "Four tasks pass every test and still score zero" | Not a finding. I read the verifier output of the current arm to explain losses measured in an earlier one; in the current arm those four score 1.0 |

Seven of those are explanations the data rejected; the eighth was a
comparison error of my own, which is the more useful of the two kinds. The
pattern across all eight is the point: what I read out of trajectory
statistics has not once survived being checked, so it is not a basis for
changing code.

Of the 21 tasks claude-code wins and pi loses, 12 are "worked and lost" — the
agent ran, produced an answer, and the answer was wrong. No mechanism in the
harness accounts for those, and seven attempts to find one have failed. The
remaining 9 are the harness failures the changes above target.

## The method that produced this

Every number here is a paired, same-task comparison with a sign test, never a
difference of totals. Arms run concurrently, because two arms in different
load windows produced p=0.041 that collapsed to p=0.453 when run together.

Three rounds were discarded before any number was reported, each for a reason
visible only inside a running container: a bundle that carried one of the
changes rather than all of them; environment variables set on the runner,
which harbor never forwards; and a marker written on the same line as the
exports it marked, which made the whole line a comment. All three logged
success.
