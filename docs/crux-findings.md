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
| "pi's read tool fails 43% of the time" | Not a finding. `isError` is a field on every one of pi's tool results and I was keyword-matching result *text* instead: the hits were successful reads of files containing the word -- an nginx config with `error_log`, a TLS-checking script, a MIPS VM. The real rate is 0.7% |
| "Four tasks pass every test and still score zero" | Not a finding. I read the verifier output of the current arm to explain losses measured in an earlier one; in the current arm those four score 1.0 |

Tool failure rates, measured from `isError` rather than guessed at: bash
9.8%, edit 5.7%, write 1.8%, read 0.7%. edit is healthy, which is why
codex's formally-specified `apply_patch` -- a real difference between the two
harnesses -- is not the thing to port here. codex's `get_context_remaining`
disqualifies itself the same way: pi's peak context is 14% of the window at
the median, so a tool for asking how much is left would always answer that
there is plenty.

Eight of those are explanations the data rejected; three were
comparison error of my own, which is the more useful of the two kinds. The
pattern across all eight is the point: what I read out of trajectory
statistics has not once survived being checked, so it is not a basis for
changing code.

Of the 21 tasks claude-code wins and pi loses, 12 are "worked and lost" — the
agent ran, produced an answer, and the answer was wrong. No mechanism in the
harness accounts for those, and seven attempts to find one have failed. The
remaining 9 are the harness failures the changes above target.

## The one difference not yet tested at its real size

pi's core system prompt is 4,169 characters of literal text. Claude Code's is
27,960 — 6.7x — and structured into sections (`# Doing tasks`, `## Bias toward
action`, `## Be concise`, `## Pacing`, `# Environment`), where pi's carries no
section structure at all.

What was tested was appending 300 characters of it. That arm is running at 1
gain and 3 losses, and it would be wrong to read that as "the prompt is not the
difference": 4,169 + 300 against 27,960 is not the comparison. Nine prompt
sections have now landed inside the noise here, and every one of them was a
few hundred characters bolted onto a prompt an order of magnitude smaller than
the one it was being compared against.

That leaves the prompt as the largest untested difference, and the only one
still standing after ten mechanism hypotheses. Testing it properly means
porting the structure, not a paragraph of it.

## The ten tasks that decide whether 80 is reachable

Over 89 tasks and thirteen arms, 80 have been solved by at least one scaffold
and 9 by none. So the ceiling on this model is 89.9%, and 80% is roughly "solve
everything that has ever been solved".

Ten of those are solved by other scaffolds and never by pi. Read one at a time,
four of them turn out not to be about solving anything:

**`pytorch-model-recovery` — a command line.** Its task text is a markdown
list, so the instruction begins with a hyphen. harbor builds `pi --print ...
'<instruction>'` with no terminator and pi's parser answers:

    Error: Unknown option: - You are given a PyTorch state dictionary

Every pi trial on it died before taking one action, in every arm, while six
other scaffolds solved it -- claude-code in eleven actions. pi already honours
`--` (`cli/args.ts:82`); nothing was passing one.

**`circuit-fibsqrt`, `write-compressor`, `schemelike-metacircular-eval` — half
a design.** Claude Code's token-budget section pairs a low output ceiling with a
clean retry at the model's own ceiling for the <1% that truncate. I shipped the
ceiling (16,384, chosen from this deployment's p99 of 16,161) and left the
retry switched off: `PI_ESCALATE_SPENT_CEILING` was never passed. What the
trials then look like:

    stop=length  out=16384  think=48437ch    no tool call
    stop=length  out=16384  think=46931ch    no tool call
    stop=length  out=16384  think=45622ch    no tool call
    stop=length  out=16384  think=53998ch    run ends

Four consecutive turns spending the whole ceiling on reasoning and taking no
action. I had been reading `stopReason: "length"` rising from 96 to 241 as
evidence the ceiling was working. It was evidence of the harm: the recovery
that makes a low ceiling safe never fired once.

The remaining six -- `install-windows-3.11` at 136 actions,
`winning-avg-corewars` at 135, `path-tracing-reverse` at 50,
`pytorch-model-cli` at 46, `cancel-async-tasks`, `gcode-to-text` -- ran, worked,
and answered wrong. Those need reading task by task.

## What 89 tasks can and cannot detect

This should have been the first calculation, not the last.

Two runs of one configuration disagree on 15-16% of tasks here, so a null
comparison over 89 paired tasks produces about 14 discordant pairs. A sign test
on 14 pairs reaches p<0.05 only at a 12-2 split:

    discordant pairs   split needed   net effect it represents
                  10            9-1                 9.0 points
                  14           12-2                11.2 points
                  20           15-5                11.2 points

**So this design can only detect an effect of nine points or more.** Every
effect measured in this project is smaller than that:

    seven source changes      +7.6 points   (8-2,  p=0.109)
    587-character prompt      -2.3 points   (2-3,  p=1.000)
    64K output ceiling        -6.3 points   (1-2,  p=1.000)

Which means "p=0.109, not significant" and "the change does not work" are
different statements, and this project has been using them interchangeably. The
seven changes may be worth seven points or nothing; 89 tasks cannot tell.

It also makes one earlier decision wrong. A replication was stopped on the
reasoning that the gains split evenly between pairs the change touched and
pairs it did not, so "more samples will not change the attribution". The
attribution argument stands, but the score question was never answered, and
stopping the replication guaranteed it would not be.

The claude-code gap is the one measurement this design can carry: 17 points is
well above the floor, which is why p=0.006 there means what it says.

The current run answers the question properly -- the same comparison at
`--attempts 3`, 267 paired observations, which brings the detectable effect
down to five or six points.

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
