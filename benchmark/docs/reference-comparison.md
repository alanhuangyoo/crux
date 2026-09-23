# Against the reference agents: what was ported, and what the corpus ruled out

Each design below was taken from a reference agent — Claude Code, Codex,
opencode, hermes-agent, grok-build — and checked against crux's own trajectories
before anything was written. The corpus is the four arms of the winning
configuration: `c16`, `c16b`, `cfin` and `cedit`, 356 trials of Terminal-Bench
2.1 on Qwen3.8-27B-FP8 at a 262,144-token window, 16 concurrent, a 120-minute
budget.

None of the adopted changes has been measured end to end; no deployment was up
when they were written. Every one of them has a unit test that fails without it.

## Adopted

| Change | Reference | What the corpus showed |
|---|---|---|
| A turn cut off while thinking gets its reasoning back | Claude Code's "pick up mid-thought", which only works where the thought survives | 292 turns ended thinking at the output limit, in 60 trials; 43% were followed by another. Reasoning is not replayed, so each retry re-derived the same plan. Nine trials ended on it with 29–83% of their budget left. |
| A turn that stops inside its reasoning is a stall, not an answer | hermes-agent's reasoning-only stall handling | 28 turns stopped (not cut) with reasoning and nothing else, in 14 trials. Three runs ended on one — `gpt2-codegolf` at minutes 1, 2 and 24, its deliverable never written. |
| A tool call cut off at the limit earns the raised ceiling too | Claude Code escalates whatever the cut response held; hermes-agent retries a cut tool call with a boosted `max_tokens` | 42 tool calls were cut, all 42 at the initial 16K with the model's own 32K unused. 15 took two or more turns, up to 32, to get the same tool through. |
| A truncated command keeps its first lines as well as its last | Claude Code keeps the first 30,000 bytes; Codex cuts the middle | 71 outputs were truncated, in 23% of failed trials against 6% of solved; 20 were followed by the model going back for what was cut. |
| A task run as one turn gets the full compaction summary | Claude Code and Codex summarize history with one full prompt | Every cut inside a single-turn run took the short turn-prefix path: 4 of 9 compactions kept 2,302–4,443 characters of ~250,000 tokens, against 7,113–13,193 through the full prompt. |
| The model is told its reasoning is not kept | Codex's tool-call preambles; Claude Code's short progress updates | Visible text beside a tool call: median 0 characters over 25,871 turns. Reasoning beside it, dropped from every later request: median 1,016 (solved) and 1,349 (failed) characters. |
| A verifier that never ran is not the agent failing | — (measurement) | 14 zeros were the verifier's own runner failing to install. `qemu-startup` and `qemu-alpine-ssh` cannot score in this environment at all. `cedit` read as a regression, p = 0.057, through a GitHub outage; with those excluded, p = 0.73. |
| The time budget follows harbor's limit for the task | — (harness) | harbor allows 80 to 1,600 minutes per task at 8×, not 120. 19 failed trials stopped themselves at 120 on tasks allowing 160–960. A number is now capped at the limit; `PI_TIME_BUDGET_SEC=task` takes all of it. |

## Measured and rejected

| Design | From | What the corpus showed |
|---|---|---|
| Repeated-identical-call detection | opencode (doom loop) | Three identical calls in a row: 0% of trials, solved or failed. |
| Answer an unchanged file's re-read with a stub | Claude Code | Exact re-reads with no edit between: about 0%. |
| Fuzzy edit matching — trimmed lines, indentation, escapes | opencode's replacer cascade | Of 111 "could not find" edits, the next successful edit of the same file differed only in whitespace once. 77% targeted text that did not exist. |
| A two-minute default command timeout | Claude Code | 161 commands with no timeout ran 2–10 minutes and finished, 99 of them in solved trials, against 60 killed at ten. After a timeout the model re-ran with a longer one 2 times in 111. |
| Reword the timeout message | — | After a timeout the model changed approach 93 times in 111; the message was not steering it into longer waits. |
| Replay each turn's reasoning | Claude and DeepSeek interleaved thinking | Failed trials' median peak context would go from 69,937 to 191,573 tokens, p90 to 691,382; 90 trials would pass 80% of the window. |
| Diagnostics after every write | grok-build, opencode (LSP) | 1,914 Python files written or edited; 9 syntax errors surfaced on the next run. |
| Repair aliased or string-encoded tool arguments | hermes-agent, opencode | 16 validation failures in 356 trials; no aliased parameter names. |
| Environment details in the system prompt | Claude Code, Codex | 0.33 turns per trial spent only on probing; probes ride along with real work. |
| "`cedit`'s edit hint lowered the score" | — | 10 of its 11 losses never produced a failed edit, and six were the verifier outage above. |

## Needs a live arm

- **Yielding long commands back to the model**, as Codex does after at most 30
  seconds. It would free the 60 ten-minute waits, and turn the 161 long commands
  that finish into polling loops for a model that thinks before every turn.
- **Longer budgets for long tasks.** `PI_TIME_BUDGET_SEC=14400` gives every task
  up to four hours within harbor's limit; it lengthens a full run too.
- **The completion notice's statistics** were measured at 32K: "6–33%" for
  solved trials, "24–95%" for failed. On this corpus they are 1–66% and 5–100%
  (p10–p90). The notice fires on most trials, so its wording is a behaviour
  change and belongs in an A/B, not an edit.

## Within a task: what separates the run that solved it from the one that did not

Comparing solved against failed across tasks mixes the scaffold with the task:
hard tasks time out more, truncate more and run longer whatever the agent does.
The cleaner comparison is inside one task. 32 tasks were solved in some of the
four arms and failed in others, under the same configuration. For each, the
failed runs' mean minus the solved runs' mean, and a sign test over the 32:

| Metric | failed > solved | failed < solved | p |
|---|---:|---:|---:|
| completion notices received | 6 | 26 | **0.001** |
| minutes used | 21 | 11 | 0.110 |
| reasoning per turn | 21 | 11 | 0.110 |
| share of tool-call turns with a note | 11 | 20 | 0.150 |
| edit failures | 8 | 4 | 0.388 |
| turns cut off while thinking | 3 | 6 | 0.508 |
| bash timeouts | 13 | 10 | 0.678 |
| tool errors | 17 | 14 | 0.720 |
| truncated command outputs | 6 | 5 | 1.000 |

No scaffold mechanism separates them. The one variable that does is a
consequence rather than a cause: a notice is only sent when the agent stops
with most of its budget left, and a run that has solved the task stops early.
Every failed run that stopped without one had spent at least 82.8% of its
budget, most of them 94–140%, and all twenty were long tasks run out of time --
train-fasttext four times, schemelike-metacircular-eval three, on limits of
240 minutes and more. What is left inside a task is which way a technical
judgement went on that run, and the time it was given to go the other way.
