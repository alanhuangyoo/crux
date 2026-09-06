# What the failures actually are

Everything here is measured on this deployment: Qwen3.8-27B-FP8 on four H20s,
Terminal-Bench 2.1 (89 tasks), SWE-bench Verified (a random 100 of 500, seed
20260906) and SWE-Atlas-QnA (124). Numbers that come from a partly finished run
say so.

Read the [resolution](#resolution) section before any comparison below.

---

## The chain

Five measurements that turned out to be one thing.

**1. What is left after the harness faults are fixed is not "the model cannot
do it".**

The three parser fixes took parse failures from 1269 to 17 and zero-command
steps from 23% to 4%. Of the 17 failures remaining in the fixed arm, the shape
is:

| | | |
|---|---:|---|
| green light, more than half the budget unspent | 10 | 59% |
| green light, less than half unspent | 2 | 12% |
| no green light, ran to the wall | 3 | 18% |
| no green light, stopped early | 2 | 12% |

**2. The green light carries no information.**

`crux submit` prints "all N item(s) verified" only after re-running every check
the agent bound. It printed that on 95% of the runs that scored and on 100% of
the runs that did not. As a test of whether the work is right, its likelihood
ratio is about 1.

**3. Bound checks almost never fail.**

88% of trials never see a bound check go red — including the ones that solved
the task. A check that has never failed has never demonstrated it can tell
right from wrong.

**4. The checklist is written after the work.**

The first check is bound at 75-87% of the way through a run, *after* the last
edit. Across 98 trajectories, **not one** bound a check before its first edit.
A checklist written afterwards describes what was built, and a description
cannot fail — which is 3, which is 2.

**5. So the agent stops early, on its own green light, with the budget unspent.**

Fraction of its own wall-clock budget used when it stopped:

| | solved | failed | ran out of time |
|---|---:|---:|---|
| TB 2.1 baseline | 12.4% | 54.9% | 10 of 27 |
| TB 2.1 fixed | 16.3% | 33.1% | 3 of 17 |
| SWE-bench Verified | 5.1% | 10.1% | 0 of 3 |

Most failures are not timeouts. They are voluntary stops with two thirds of the
budget unspent, and on SWE-bench with nine tenths.

### Two tasks that show it

Both are tasks claude-code solved and crux did not.

**sanitize-git-repo.** crux bound `grep -rq '<your-aws-access-key-id>' .` — a
case-sensitive search for the placeholder it had just substituted in. The
grader ran the same idea lowercased, over text the agent never looked at, for a
`FAKE_HUGGINGFACE_TOKEN_2` it had never heard of. crux submitted after four
minutes of a two-hour budget; claude-code took nine and got it right.

**cancel-async-tasks.** crux bound its own `test_a.py`, `test_b.py` and
`sigint_test.py`. The grader asserted `stdout.count("Task started.") == 2`
under concurrency, which none of those scripts exercised.

Same shape: the check inherits the implementation's misunderstanding, because
the same reasoning wrote both.

---

## What was built on it

| mechanism | what it does | state |
|---|---|---|
| verification-debt gate | blocks the 13th edit made without running anything that checks it | on, threshold 12 |
| bind-time probe | runs a check when it is bound and says if it already passes | on |
| budget notice | at "are you sure you are done", how much of the run is left | on |
| submit gate | `crux submit` refuses once and asks for one more check from the task's own words | off by default, an arm |
| checklist-first gate | holds the first edit until one check is bound | off by default, an arm |

The gate threshold of 12 is the argmax of a sweep over 89 trajectories: it
catches 62% of failures at a 15% false-fire rate. 20 would have caught 37%.

---

## Nine lines that were measured and dropped

Kept because "looked and decided against" and "never looked" are different
things, and only one of them is a result.

| line | why not |
|---|---|
| numeric-threshold assertions (`iou >= 0.5`, off by 1.4%) | 2 such assertions in 87 graders; designing for it is fitting two tasks |
| polling sleeps | 108 occurrences, all from one task |
| unexplained timeouts | 4 tasks, no shared shape |
| task-shipped checkers | 91% of tasks have none; all 3 that had one and ignored it solved anyway |
| widening the verification regex | separation 0.49 → 0.46 |
| not counting `/tmp` writes as edits | 0.49 → 0.40; scratch writes correlate with the failure shape too |
| requirement-coverage ratio | separation 0.29; the requirement count is 23 in both groups, so it adds nothing over the raw check count |
| a gate on checks naming the task's own deliverable | per-check the split is real (failed runs spend 60-70% of their checks on their own scaffolding vs 20-35%), but per-run separation is 0.11 |
| command repetition | consecutive > 2 gives 0.24, near-repeat > 1% gives 0.36, and neither is monotone — solves reach 26 consecutive repeats, failures 17 |

And one mechanism removed:

**The stuck notice.** It fired past 120 steps and told the trial its approach
had failed. 120 was measured: on the baseline no solve exceeded it and all six
trials past it failed. The parser and blocking fixes then removed the waste
those trials were made of. On 80 trials of the current code the longest solve
takes 182 steps, failures stop at 132 rather than 519, and **six of the ten
trials past 120 solve**. Crossing it now correlates with succeeding, no
threshold in the current data separates the two, and the notice was telling a
healthy trial to abandon a working approach. A measured constant is only as
current as the code it was measured on.

---

## Resolution

**Two runs of one configuration disagree on 15-16% of tasks.** Measured:
`crux-jobs` vs `crux-w-jobs`, 85 shared tasks, 14 flips, identical scores (53
vs 53); `crux-m-jobs` vs `crux-v-jobs`, 89 shared, 13 flips.

So an 89-task run resolves about **±8 points**, and a 29-task run about ±15. A
lead smaller than that is not a lead. This is why `gate-on` versus `full-fixed`
— 9 to 5 on 16 shared tasks with four disagreements all one way — was read as
the gate working and was not: the two arms have the same configuration.

**A partly finished run reads high, always.** Failed trials take about twice as
long as solved ones, so what has finished over-represents successes. SWE-bench
read 96% at 26 of 99 and 84% at 68.

**Partial credit does not rescue it on Terminal-Bench.**

| | tests per task (median) | tasks scoring between 0 and 1 |
|---|---:|---:|
| Terminal-Bench 2.1 | 3.0 | 14% |
| SWE-bench Verified | 56.0 | 18% |
| SWE-Atlas-QnA | 9.5 | 57% |

Three tests per task is no gradient. Paired on 62 shared tasks it moves the
signal from 14 tasks to 17 and the verdict not at all (78.1% vs 76.5%, 9-8,
p=1.0). On SWE-Atlas it is the right metric and the binary one is misleading:
33% of tasks pass while 90% of rubrics do.

So on Terminal-Bench the only way to resolve a five-point difference is more
runs.

---

## Three mistakes worth not repeating

**Grepping a trajectory counts the prompt.** Step 0 documents the tools, so a
file-wide grep for `crux todo add ... --verify` reports every trial as having
used it. It cost three separate readings: the prompt's regex example counted as
a bound check in three tasks, "all N item(s) verified" appearing in 100% of
both groups, and 10-of-10 usage where the answer was 0-of-10.
`crux.watch.commands_of` exists so nothing has to remember this.

**Comparing counts across arms whose runs differ in length.** Failing runs go
to the wall, so every absolute count is higher in them. Seven different signals
all returned exactly 57:1 before the confound was visible. Rates, and paired on
shared tasks.

**Verifying the wrong layer.** `docker exec` cannot see a tmux session's
environment, so a submit gate that was correctly armed read as unarmed. The
check has to be at the layer the thing runs in — `tmux show-environment`, not
`docker exec env`.
