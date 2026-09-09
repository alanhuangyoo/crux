# What this branch changes, and what each change was answering

Fourteen commits on top of v0.85.1, 1,076 lines. Every behavioural change is
taken from a shipped reference implementation rather than reasoned out, and
every one names the measurement that motivated it. The benchmark throughout is
Terminal-Bench 2.1, 89 tasks, one self-hosted Qwen3.8-27B, 8x agent budget,
arms run concurrently.

## The loop

**A turn that answers nothing is not the agent finishing.** The loop ends when
an assistant message carries no tool calls, which is right for a message that
answers and wrong for one cut off mid-sentence. `regex-chess` spent 65,536
output tokens -- 199,246 characters, entirely thinking, `stopReason: "length"`
-- and the run was recorded as settled having taken no action at all.

**Two-phase recovery, in Claude Code's order.** Its query loop handles
`max_output_tokens` as `escalate` then `recovery`, and the order was the part I
had backwards: a truncated turn usually means the model needed more room, so
raise the ceiling and retry *silently* first; only tell it when the raised
ceiling is spent too, bounded at three. Scoped to a turn that spent the ceiling
exactly -- the other case pi already recovers a layer up by compacting and
retrying, and taking it away broke three of its characterization tests.

**A deadline the loop can see.** `AgentLoopConfig` had no notion of time. A
harness enforces one from outside, so a run ends mid-tool-call with whatever
was on disk at that instant. Six of pi's twenty-four failed trials ended that
way with work in flight, while the median solve used a fraction of its budget.

**Why the loop continued, recorded rather than inferred.** Claude Code carries
this on its State object as `transition`. Emitting it on `turn_end` answers a
question that was otherwise unanswerable from outside -- whether a recovery
fired at all. Two rounds of A/B attribution had to be done by grepping trial
logs for notice text, which finds nothing for a recovery that prints nothing,
and the silent escalation is exactly that case.

**One state value, not four flags.** `escalated`, `outputLimitRecoveries`,
`deadlineWarned`, `unactionableTurns` -- three added by this work, each shaped
to suit the change being made, none readable from outside the function. That is
how a loop grows a fifth flag. `LoopState` collects them, rebuilt immutably at
each point that spends a recovery, the way Claude Code rebuilds its State.

## The tools

**A bash command the model did not bound is still bounded.** `timeout` is
optional and documented as having no default: across 6,100 bash calls, 85.7%
carried none, and 13 of 401 trials ended on a tool call that started and never
finished. Claude Code answers this with a tiered policy; this takes its 120s
default, since `MAX_TIMEOUT_SECONDS` already serves as the ceiling and pi
reports a timeout as an ordinary tool error the model can read.

**A ceiling on one response, with the retry that makes it safe.** A provider
reserves inference capacity by `max_tokens`, so a large one costs queueing
whether or not the tokens are used. Claude Code caps at 8K against a p99 output
of 4,911 and gives the truncated <1% a clean retry. Over 10,225 turns here p99
is 16,161, so the same method gives 16K, which truncates 0.4%. A setting, not a
new default: the number belongs to whoever runs the model.

## What the measurements ruled out

Twelve candidates for a larger change, each checked before writing code, and
none of which survived: sub-agents (0 of 89 claude-code trials spawned one),
the todo tools (69.7% against 66.7% controlling for run length), MicroCompact
(pi's peak context is 14% of the window at the median), token-budget nudges
(they exist to stop an agent sooner; pi already stops too soon), Grep and Glob
(0 calls in 89 trials), read deduplication (pi repeats 4.6% against claude
code's 7.5%), codex's `apply_patch` grammar (pi's edit fails 5.7%), codex's
`get_context_remaining` (would always answer "plenty"), and Claude Code's
blocking stop hook -- which pi already has, as `getFollowUpMessages` fed by the
queue `sendUserMessage` writes to.

Three more died as measurement errors of mine rather than as findings,
including a claim that pi's read tool fails 43% of the time. `isError` is a
field on every one of pi's tool results; I was keyword-matching result text,
and the hits were successful reads of files containing the word. The real rate
is 0.7%.

## What this branch does not show

It does not show these changes are worth points. 79 paired tasks gave 8 gains
and 2 losses at p=0.109, and the gains did not concentrate where the change
acted.

The reason that is not a verdict is a calculation that should have come first.
Two runs of one configuration disagree on 15-16% of tasks here, so 89 paired
tasks give about 14 discordant pairs, and a sign test on 14 reaches p<0.05 only
at 12-2 -- a net effect of 11 points. **This design cannot detect anything
smaller than nine points, and every effect measured here is smaller than that.**
"Not significant" and "does not work" are different statements.

A run at `--attempts 3`, 267 paired observations, is what answers it.
