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

### The same shape on a different benchmark

SWE-bench Verified finished: **73 of 89 scored, 82.0%**, on a random 100 of the
500 (seed 20260906; 10 trials were lost to an operator error and their absence
moves the number by 0.2 points). Official swebench 4.0.3 grading, which resets
the agent's edits to the test files before applying the real test patch. Median
trial: 18 minutes against a 400-minute budget.

Its 16 failures are the chain again, harder:

| | | |
|---|---:|---|
| green light, more than half the budget unspent | 15 | 94% |
| green light, less than half unspent | 0 | |
| no green light, ran to the wall | 0 | |
| no green light, stopped early | 1 | 6% |

**Not one failure was a timeout.** And 15 of the 16 missed by two tests or
fewer:

    pydata__xarray-4687      1717/1718 tests   green   3.2% of budget used
    astropy__astropy-14369    733/735          green  10.1%
    django__django-13513       83/84           green   5.1%
    django__django-15987       52/53           green   2.9%
    django__django-13512       33/35           green   2.1%

The first line is the whole argument in one row: 1717 tests passed, one failed,
the agent declared itself finished on its own green light and stopped with 96.8%
of its budget unspent.

One thing differs from Terminal-Bench and is worth noting, because it rules
something out: **the number of bound checks does not separate here** — solved
and failed trials both bind a median of 4. The problem is not verifying too
little. It is verifying the wrong thing.

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
| `crux tests` | reads the repository's own test invocation out of its CI config | on, in the prompt |
| `crux falsify` | empties the file a check names and reports whether the check notices | implemented, replayed as ineffective here |

The gate threshold of 12 is the argmax of a sweep over 89 trajectories: it
catches 62% of failures at a 15% false-fire rate. 20 would have caught 37%.

---

## Ten lines that were measured and dropped

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
| vacuous bound checks (`crux falsify`) | replayed offline against every bound check in both finished runs: the shapes it would call vacuous are 5% of checks in TB solves and **0% in either set of failures**, and failures read the deliverable's contents *more* often than solves (94% vs 86%). `mteb-retrieve`, whose whole verification was that a file existed, is the exception and not the shape. Three minutes of replay instead of a GPU run |
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

## The variance nobody set

crux has never set a sampling temperature. Terminus passes one only when it is
explicitly configured, and crux never configures it, so **every number this
project has produced was sampled at the server default** -- the
maximum-variance setting. Measured on the endpoint, same prompt five times:

    no temperature set (what every run used)   3 different answers
    temperature=0                              1 answer

That is the likeliest single explanation of everything variance-shaped here:

| | |
|---|---|
| 60% | of SWE-bench failures solve on a plain re-run, nothing changed |
| 15-16% | of tasks flip between two runs of one configuration |
| ±8 points | is all an 89-task run can resolve |

It also explains why both gates looked effective. In a system this noisy,
anything selected on failure looks better when it is run again.

And it puts a number on the headroom. On SWE-bench Verified: pass@1 is 82.0%,
and 9 of the 15 failures re-run solve, so pass@2 is bounded at 92.1%. **The
tasks this agent genuinely cannot do are 6 of 89 -- 7%.** The other 11% is
sampling.

So the lever is not more verification. It is less variance.

**And the obvious knob pulls the wrong way -- because there was nothing wrong
with the setting.** The framing this experiment started from was that crux had
never set a temperature and therefore ran at the highest-variance setting.
Reading the model's own files says otherwise:

    generation_config.json:  temperature 1.0, top_k 20, top_p 0.95
    sglang:                  sampling_defaults='model'

and the model card's recommendation, verbatim:

> Thinking Mode: `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`,
> `presence_penalty=0.0`, `repetition_penalty=1.0`

**The deployment already runs exactly the recommended thinking-mode sampling.**
Qwen's own SWE-bench Pro and DeepSWE 1.1 numbers are reported at temp=1.0 and
top_p=0.95 through the Claude Code harness -- the same settings measured here.

So `temperature=0.2` was not tightening a loose knob, it was departing from the
vendor's recommendation, and the result is what the card warns about. The same
16 tasks a third time at `temperature=0.2`, against the two arms at the default:

| | consecutive repeats (median) | repeated commands | most-repeated command |
|---|---:|---:|---:|
| temperature 0.2 | 2 | 14.7% | 22x |
| server default | 1 | 4.7% | 7x |
| server default, gate on | 1 | 5.5% | 9x |

Command repetition triples, and the first four tasks scored 0 against 8-of-14
and 9-of-15 at the default. This is the failure Qwen3's own card warns about:
low temperature in thinking mode falls into repetition loops. Lowering the
temperature does not remove the variance, it trades sampling variance for
loops.

So the 60% re-run recovery and the 92.1% pass@2 bound both stand, and this is
not the way to reach them.

---

## What the SWE-bench failures are, exactly

Sixteen failures, traced one at a time. Each step below killed a hypothesis.

**They are regressions, not missing features.** Classifying each failure by
whether the test it failed existed at the base commit:

| | |
|---|---:|
| broke a test that already existed | 11 |
| missed a behaviour the test patch adds | 2 |
| both | 3 |

**It is not that the agent skips testing.** It runs the repo's own suite a
median of 7 times when it solves and 10 when it fails; 88 of 89 trials run it
at least once. As a separator that is worth 0.06.

**It is not that it edits after testing.** 71% of solves and 69% of failures
run a test after their last edit.

**It is not the wrong runner.** Every django trial uses `runtests.py`, 39 of 41
solves and 8 of 8 failures. None reaches for pytest on a repo that needs
django's own harness.

**It is not testing too narrow a scope.** For 11 of the 16, the module holding
the broken test is one the agent ran.

**It ran that module and its run passed.** `django__django-16263` ran a
1243-test regression sweep -- `Ran 1243 tests in 6.554s / OK` -- and the grader
still failed it on `aggregation.tests`. Same module, same code, opposite
result.

**The difference is the invocation.** The grader runs

    ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 <modules>

and of the eight django failures, **none used `--parallel`** and only two used
`--settings`. Of the 41 solves, six used `--parallel`. Different settings module
and different test isolation, so the same tests are not the same tests.

That points somewhere the gates do not. How the grader runs the suite is not a
judgement the agent has to make about its own work -- it is a fact in the
repository, in CI config, tox.ini, or CONTRIBUTING. It is an external signal,
which is the thing every mechanism in the section below turned out to lack.

---

## Neither gate survives attribution

Both gates were tested with a control arm, and both looked like they worked.
Neither does.

**The submit gate, on SWE-bench.** Sixteen tasks the benchmark had failed, all
of them a green light with the budget unspent, re-run with `crux submit`
demanding a second pass. Six of eleven flipped, and every flip went from
missing one to three tests to passing all of them:

    django__django-11433     141/143  ->  143/143
    django__django-14155      89/92   ->   92/92
    django__django-15987      52/53   ->   53/53
    django__django-16263     102/103  ->  103/103
    sphinx-doc__sphinx-9229    13/14  ->    14/14
    sympy__sympy-13878         19/20  ->    20/20

A perfect shape. Then: **the gate fired on 5 of 15 trajectories, and on only
one of the six that flipped.** Where it fired, 1 of 3 solved; where it did not,
5 of 8. The tasks were selected because they failed, so a re-run recovers some
of them by regression to the mean alone -- and that is what this is.

**The edit-debt gate, on Terminal-Bench.** The 29 tasks the baseline failed,
gate on against gate off, verified as firing 8 times in one arm and 0 in the
other. 9 of 16 against 6 of 16, three disagreements all one way, tests passed
65.6% against 50.0%. Then, split by whether the gate fired:

|  | gate on | gate off |
|---|---:|---:|
| the 7 tasks it fired on | 4/7 | 3/7 |
| the 9 it did not | 5/9 | 3/9 |

Most of the gain is on tasks the gate never touched, and one of the three flips
had it fire. Two flips are what noise predicts on 16 tasks.

**Both are score-supported and mechanism-refuted.** Reading only the scores,
tonight produces two validated mechanisms. Adding one question -- *did the thing
actually fire on the tasks whose outcome changed* -- and both fall over. The
question costs a grep of the observations, and it is the difference between a
result and a story.

---

## The ceiling

The same 16 SWE-bench tasks were run four times -- once in the main run and
three more with the submit gate on, off, and at a lower temperature. Across all
four:

| | | |
|---|---:|---|
| solved on the first attempt | 73 | 82.0% |
| solved in some run but not the first | 12 | 13.5% |
| never solved in any of four runs | 4 | 4.5% |

**What this agent genuinely cannot do is 4 tasks of 89.** Its ceiling is 95.5%,
it sits at 82.0%, and the 13.5 points between are sampling.

Which is why every mechanism tried tonight failed, and why they failed the same
way. There is nothing wrong with the work that more checking would find; the
same agent, given the same task again with nothing changed, does it correctly
60% of the time. The problem is not that it verifies too little. It is that the
run it happened to take was one of the bad ones, and it had no way to know.

Reducing that needs one of two things:

  * **less variance at the source.** Tried: temperature 0.2 triples command
    repetition and scores 2 of 12 against 9 of 15. Qwen3's card warns of
    exactly this in thinking mode, and it is what happened.
  * **a signal that a run went badly.** Everything tried here asks the agent to
    judge its own work, and its green light has a likelihood ratio of 1.

The one candidate that produces a signal without asking for a judgement is
**disagreement**: solve it twice and compare. On these 16 tasks, four runs
disagree on 11 of 15 -- and disagreement is exactly the set that needs more
work. It costs a second pass, which the budget already has: failures stop with
two thirds of their wall-clock unspent, and on SWE-bench with nine tenths.

It is not implemented here. It changes the agent's main loop rather than adding
a command, and the honest state of it is: measured as promising, not measured
as working.

---

## Where the actions go

Measured across both finished runs and claude-code's, with no GPU:

| | commands / tool calls | file operations | through a structured tool |
|---|---:|---:|---:|
| crux, TB 2.1 | 6,956 | 70% | 0% |
| crux, SWE-bench | 6,826 | 83% | 1% |
| claude-code, TB 2.1 | 4,013 | 18% are Read/Edit/Write/Grep | — |

Paired on the 72 tasks both scored, in units that are the same on both sides:

| | steps (median) | tool calls (median) | calls per step |
|---|---:|---:|---:|
| crux | 50 | 81 | 1.55 |
| claude-code | 27 | 29 | 1.06 |

**1.85x the steps and 2.8x the tool calls, on 92% of the tasks, for the same
score.** Reading a file alone is 41-49% of every command crux issues: `cat`, `head`,
`sed -n '50,100p'`, a slice at a time. **claude-code finishes the same 89 tasks
in 4,013 tool calls against crux's 6,826 -- 70% fewer -- for the same score.**

`crux read/grep/edit/write` are installed and the prompt turns them off. The
comment on that decision:

> the model does not take them up: over 206 tool calls in one evaluation,
> read/grep/edit/write together accounted for under 2%

That was measured on the code that still had the XML parse faults, and the same
thing has already happened once tonight: the stuck notice's threshold of 120
was correctly measured and then invalidated by the parser fixes, which
collapsed the failure tail from 519 steps to 132.

Then it turned out the switch was not reachable at all.
`build_terminus_template` has no `file_tools` parameter, so the Terminus prompt
never named `crux read`, `crux grep`, `crux files`, `crux edit` or `crux write`
-- while `_install_crux` put all five in every container. **0% uptake was a
statement about the prompt.** The binaries were installed and never mentioned.

So the chain reads:

    the prompt never named the file tools
      -> 70-83% of actions go through the shell, a slice of a file at a time
      -> 1.85x the steps, 2.8x the tool calls
      -> a longer trajectory
      -> more variance
      -> the 13.5 points between 82.0% and the 95.5% ceiling

Every link is measured except the last. Whether fewer actions actually reduce
the variance is the question a run answers, and it is now reachable:
`--agent-kwarg file_tools=1`. Off by default until it is measured.

**What the arm shows so far, and what it kills.** Naming the tools moved uptake
from 0.3% of commands to 4.3% -- fourteen times more, and still small. Shell
reads barely moved, 42.1% to 40.0%, and 14 of 29 trajectories never touched a
structured tool at all. The usage that did happen is correct: `crux read
/app/decomp.c --limit 200`, `crux edit` with a well-formed JSON body.

The obvious next move was to rewrite the prompt's examples, on the theory that
the model copies examples rather than following advice. The prompt says
otherwise:

| | mentions | first appears at |
|---|---:|---:|
| `cat` | 2 | 11% |
| `head` / `tail` | 0 | — |
| `grep` | 2 | 89% |
| `sed` | 4 | 84% |
| `crux read` | 1 | 89% |
| `crux edit` | 2 | 93% |

**The prompt is not pushing the shell.** It mentions `cat` and `sed` about as
rarely as it mentions `crux read`, in the same last tenth of the text. The
model reaches for `cat` 773 times because that is its prior, not because it was
told to. A section saying "prefer these" moving uptake from 0.3% to 4.3% is
what one section can do against a strong prior.

So the v2 that rewrites examples has no evidence behind it. What would change
the behaviour is making the shell path unavailable or the structured tool the
only route -- a much larger change, and worth attempting only if this arm's
score says the actions matter.

---

## Trials that stop, and the slot they hold

Across 524 trials in twelve runs, 28 stopped without ever finishing -- no
verifier output, and a trajectory that ends early and is never written to
again. They are not spread evenly:

| task | stalled | seen | rate | steps when it stopped |
|---|---:|---:|---:|---:|
| regex-chess | 6 | 11 | 55% | 3 |
| adaptive-rejection-sampler | 4 | 10 | 40% | 6 |
| dna-assembly | 3 | 11 | 27% | 11 |
| fifteen others | 1 each | | | |

Three tasks account for 13 of the 28, and all of them stop early rather than
part-way. Each stall holds a concurrency slot for about 100 minutes, so 28 of
them is roughly 47 slot-hours.

**What it is not.** Four explanations were checked and none of them holds:

- *the model call hanging without a timeout* -- `timeout` does reach litellm
  and does work: 5 seconds gives a Timeout in 5.2s, and no timeout hangs
  forever. The 600s limit crux now sets is real, and these trials outlast it.
- *the container waiting on a long command* -- inside a stalled container the
  only processes are `sleep infinity`, tmux, and an idle `bash --login`.
  Nothing is running.
- *resource limits* -- the stalling tasks have exactly the config of tasks that
  never stall: 1 cpu, 2048 MB.
- *the OOM killer* -- two containers were OOM-killed across the whole set, and
  neither is one of the three.

**What it is.** The pane, the asciinema recording and the trajectory all stop
being written in the same minute, and the container goes idle. That places it
in harbor's own coroutine rather than in the model call or the container. It is
not diagnosed further here.

One thing worth carrying: `regex-chess`'s image has no `ps`. It is a stripped
image, and it is the worst offender at 55%. That is a correlation with one
sample of a stripped image, not a finding.

---

## Reading the other agent's trajectories

claude-code's full trajectories for all 89 tasks have been on disk the whole
time and had only ever been counted, never read. Read side by side on
`chess-best-move` -- a task it solved in 36 steps and crux failed in 55 -- three
differences looked obvious:

| | claude-code | crux |
|---|---|---|
| step 1 | `Read /app/chess_board.png` | `ls -la /app` |
| dependency probes | one command | three steps |
| step 5-10 | already scanning 64 squares | `pip install`, then four idle steps, then `C-c` |

Two of the three did not survive the corpus.

**Installing dependencies.** crux does it *less* than claude-code -- 1.0% of
commands against 2.9% -- and across the whole run "install then idle" happens
for a total of one step. The four idle steps in that trajectory are one task's
behaviour, not a shape.

**Reading the artifact first.** One task. Not measured further.

**Idle steps survived, and are not what they looked like.**

    claude-code    1 idle command out of 2,939     0.0%
    crux         176 idle commands out of 7,151    2.5%

But 176 steps in 147 runs, median length 1: not one long wait for an install,
but a hundred and forty-seven single steps that send nothing in order to look
at the terminal again. What precedes them is `crux submit`, `crux todo list`,
`C-c`, a slow query.

Terminus's protocol allows an empty keystroke as a way to refresh terminal
state. claude-code's tool protocol has no such move -- every call must be a
Bash, a Read, an Edit. **The difference is not judgement, it is whether the
protocol offers a way to spend a turn on nothing.**

Worth 2.5% of commands. Recorded because the method is right even when the
lever is small: the other agent's trajectories answer questions about our own
that no amount of reading our own code will.

---

## The output cap was the wrong cause

The chapter above reads claude-code's `regex-chess` trajectory, finds

    API Error: Claude's response exceeded the 64000 output token maximum.

and concludes that runaway generation is what stalls these trials. A 32k cap
went in on that basis. It works and it changes nothing.

    completion tokens     over 32,768
    all-on (old code)     9 of 4,938 steps
    ft6, think6, both2    0 of 1,396 steps

The cap is in force -- the old arms exceed it, the new ones never do -- and the
same three tasks stall for the same 45-48 minutes.

**The second guess was also wrong.** Reading further, claude-code's stalled
steps take 53-60 minutes each and issue *zero tool calls*: the model is
thinking and never emitting a command. That looked like the answer, so: does a
long reasoning block predict a step with no command?

| reasoning length | steps | no command |
|---|---:|---:|
| 0-2k | 2,992 | 1.7% |
| 2-6k | 771 | 2.1% |
| 6-12k | 513 | 0.2% |
| >12k | 662 | 2.9% |

No separation. And the single longest reasoning block in the corpus -- 138,617
characters -- is in a trial that *scored*.

What survives is one observation, not a cause: claude-code stalls on the same
tasks crux stalls on, spending 50-60 minutes per step with no tool call, and
ending in an API error that crux has no equivalent of. Same behaviour, and
only one of the two is instrumented to notice.

Both the cap and the timeout are worth keeping -- an unbounded generation and
an unbounded request are real hazards whatever else is true. Neither is the
cause of this.

**Two more eliminated.** The endpoint is not involved: during a stall it runs
22 requests at 1,365 tok/s with an empty queue and no errors, while 29
containers are up. The seven missing requests are the stalled trials -- they
never reach the server at all.

Inside a stalled container, `tmux list-panes` reports `pane_current_command` as
`python3` while no python3 process exists. That looked decisive -- Terminus
decides a command has finished from exactly this -- until the same field was
read across every live container:

    build-pov-ray          python3   (progressing)
    circuit-fibsqrt        python3   (stalled)
    train-fasttext         python3   (progressing)
    regex-chess            python3   (stalled)
    ... all 20 the same

`asciinema rec` is a python3 program wrapping the shell, so the field reads
`python3` everywhere and separates nothing.

Ten explanations checked, ten eliminated: the call timeout, container
activity, resource limits, the OOM killer, runaway generation, reasoning
length, the endpoint, the pane command, terminal encoding, and output volume.

What the stall actually looks like, stated precisely enough to hand to someone
else:

    the command completed          pane ends `\x1b[?2004h` + prompt
    the command was ordinary       `Rscript --version`, `python3 -c "import chess"`
    the endpoint has no request    22 in flight for 29 containers
    the container is idle          only sleep infinity, tmux, an idle bash
    all three output streams stop  pane, cast and trajectory, same minute

It is stuck after harbor has the terminal output and before it issues the next
model call. Ten guesses in, the list of what it is not is the only product, and
that is where this line stops.

---

## Stopping a loss you cannot fix

The stall has ten eliminated explanations and no cause. What it does have is a
price, and the price can be asked a different question: *what are these tasks
worth?*

Over 432 scored trials in 21 runs:

| task | solved | stalls |
|---|---:|---:|
| regex-chess | 0 of 4 | 6 of 11 |
| adaptive-rejection-sampler | 0 of 6 | 4 of 10 |
| write-compressor | 5 of 6 | yes |
| circuit-fibsqrt | 3 of 4 | yes |

The first two have never solved. The last two stall as well and solve most of
the time, so capping them would trade real scores for wall-clock. Only the two
with nothing to trade get a shorter budget: 30 minutes instead of 120, run in
their own job rather than excluded, because a task that starts solving under a
new configuration has to be able to show up and a quietly skipped one never
can.

That is the whole mechanism, and it is worth stating because it generalises:
**a cost you cannot remove can still be priced, and something priced at zero is
not worth an explanation.** Ten hours went into looking for the cause of these
stalls. Ten minutes of asking what the tasks score would have capped them on
the first night.

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

**Two questions that look like one.** "Which trials came back without a
score" and "what does a resume have to re-run" differ by every task the run
never reached. `unscored` answers the first; using it for the second dropped 72
of 89 tasks silently, because an 89-task run stopped at 6 scored had only
created 17 directories. `crux.watch.remaining` answers the second, from the
run's own recorded task filter.

**Verifying the wrong layer.** `docker exec` cannot see a tmux session's
environment, so a submit gate that was correctly armed read as unarmed. The
check has to be at the layer the thing runs in — `tmux show-environment`, not
`docker exec env`.

## One flag, two tasks, six days

Every trial that ran with `confirm_gate` on a qemu task died before the agent
typed a character. Not most: all of them.

    qemu trials with confirm_gate      38 / 38 dead in setup
    qemu trials without it             73 / 73 reached the agent
    non-qemu trials with confirm_gate  396 / 396 unaffected

Perfect separation over 111 qemu trials and 434 gated ones. The error every one
of them recorded:

    RuntimeError: Failed to start tmux session. Error: None

`confirm_gate` sets one environment variable inside the task container.
Terminus delivers `extra_env` as `tmux new-session -e KEY=value`, and `-e`
arrived in tmux 3.2. Both qemu images are Debian 11, which ships 3.1c:

    $ docker run --rm alexgshaw/qemu-alpine-ssh:20251031 tmux -V
    tmux 3.1c
    $ ... tmux new-session -e FOO=1 -d -s t bash
    tmux: unknown option -- e

**`Error: None` is not tmux staying quiet.** harbor builds its docker exec with
`stderr=asyncio.subprocess.STDOUT`, so `ExecResult.stderr` is structurally
always `None` and `ExecResult.stdout` holds everything — and the one line that
reports the failure formats `stderr`. `unknown option -- e`, `tmux: command not
found` and a container that died all print the same nine characters. That is
why this sat for six days looking like an environment flake: the record of 38
identical deaths contained no information at all, so there was nothing to
notice, and a task that fails in setup on every run stops looking like a
failure and starts looking like the weather.

Two fixes, in `terminus_agent.py`:

- `_route_env_around_old_tmux` feature-tests the flag — a probe session, not a
  version string, because "3.2 or newer" is a claim about a changelog — and on
  a tmux without it writes the variables to `/etc/profile.d/crux-env.sh`, which
  the session's `bash --login` sources, then clears `extra_env` so the rejected
  flag is never built. Verified in the image itself: `GATE=1` inside the pane.
  Whether profile.d is reached is a property of the image, so it is checked and
  a variable that did not arrive is logged rather than assumed.
- `_why_tmux_failed` re-runs the start command and reports what it printed, so
  the next failure of this shape names itself.

### What it cost the measurements

`analysis.py` counts an errored trial as zero, which is what a leaderboard
does. So every arm carrying `confirm_gate` — `all-on`, `submit-gate`, `ft-on`,
`ft-on2`, `think-*`, `both-*`, `json*` — ran against controls that did not,
with two tasks pre-set to zero on the treatment side only. qemu solves 47% when
it starts, so the bias is about one task in 89: −1.1 points, always in the same
direction.

That is too small to have moved any verdict here; `p=1.0` does not become
significant on one task. It is worth writing down anyway, because it is a
different animal from the noise everything else in this file is measured
against. The 15% flip rate is symmetric and averages out with samples. This
does not average out — it is a constant subtraction applied to one arm, and the
only reason it did not matter is that the effects being measured were smaller
still. A mechanism worth +2 would have been reported as +1.

### The general shape

Every mechanism in this project has been judged on a score. A mechanism can
also change whether the trial *runs*, and that failure does not look like a bad
score — it looks like infrastructure. The check that would have caught it in a
minute is not about scores at all:

    does the set of tasks that produced a score change when the flag flips?

It changed by exactly two, every time, for six days.

## The gate obeys, and it costs

`confirm_gate` was filed with the other six as "inside the noise". On score
that is right. On behaviour it is wrong, and the difference matters.

Against `base-crux`, same 89 tasks, same 8x budget, paired:

    score        59/86 -> 54/86      gains 11, loses 16    p = 0.44
    median steps    36 -> 51         more on 53, fewer on 21   p = 2.6e-4
    timeouts         9 -> 23
    setup deaths     0 ->  2         (the tmux `-e` bug above)

Steps are throughput-independent, so unlike wall-clock they are not an
artifact of what else was on the box. The gate does what it was built to do:
it refuses the first `crux submit` and the agent goes back to work. It goes
back to work on 53 of 74 tasks where the count moved at all, and it converts
nine timeouts into twenty-three — at 8x budget, two hours each.

**Six mechanisms were ignored. This one was obeyed and bought nothing.** That
is a different result and it deserves a different name. "No effect" invites
"say it louder"; every prompt section here has been a louder version of the
same request. What actually happened is that the agent did the extra pass, and
the extra pass did not find anything, because the thing it is asked to check
with is its own judgement — the same judgement that produced the answer.

So the gate comes off, and it takes the arms built on it with it. Every
mechanism measured since 09-06 used `all-on` as its baseline, which means each
was measured on top of an agent already running 42% longer with 2.5x the
timeout rate. The comparisons are internally valid — the gate is on both sides
— but the operating point is a poor one, and `base-crux` is a better control
that already exists at 89 tasks. `think-clean` and `ft-clean` re-ask the two
open questions against it, with the two qemu tasks alive for the first time.

## The stall had a message all along

Twenty-one chapters of this file eliminate explanations for the stalls: trials
that stop early and hold a slot for an hour. Ten causes ruled out, cause
unknown, priced instead of explained. Two tasks — `regex-chess` and
`adaptive-rejection-sampler` — were written down as never solving and given a
reduced budget on that basis.

The two clean arms report it in one line:

    litellm.Timeout: APITimeoutError - Request timed out.
    timeout value=600.0, time taken=1801.36 seconds

Three attempts of 600 seconds. **The stall is this project's own constant.**

`_LLM_CALL_TIMEOUT_SEC = 600` was set against a 900-second budget, where it
bounds a hung request to two thirds of a trial — litellm's own default is 6000s
and harbor never overrides it, so some ceiling was needed. Every run since has
used an 8x budget, where the same number is 8% of the trial. It stopped
bounding hangs and started cutting work.

The distribution says how narrow the cut is. Across 9,253 inter-step gaps in
two 89-task runs, 99.8% are under 600 seconds. But the five tasks that stall —
`write-compressor`, `circuit-fibsqrt`, `regex-chess`, `dna-assembly`,
`adaptive-rejection-sampler` — spend 20-30% of their steps above it. These are
the tasks whose turns run to 20,000 completion tokens; at the throughput 30
concurrent containers leave, that is ten to thirty minutes of generation. The
cut is not broad. It is aimed exactly at the tasks that were then recorded as
unsolvable.

What the trial sees afterwards is `service "main" is not running`: the timeout
kills the call, the call kills the trial, and by the time the agent's next
keystroke goes out the container is gone. That message is the one piece of
positive evidence the stall investigation never had — every earlier finding was
an elimination.

The cost, measured on the two arms: eight stalled trials, 719 minutes of
agent execution, **12 slot-hours**, on the same five tasks both times.

The fix scales the ceiling to the budget — a quarter of the trial, floored at
the old 600 and capped at 1800 — so a hang is still bounded and a ten-minute
generation is no longer thrown away. At 8x that is 1800 seconds, which is one
attempt longer than the 1801-second turn that killed `regex-chess`.

### What this reverses

"`regex-chess` 0/4 and `adaptive-rejection-sampler` 0/6 have never solved, so
they get a x2 budget in a separate job rather than x8" — that was in this file
as a decision made after ten causes were ruled out. It priced a constant of my
own as a property of the tasks. Both tasks go back into the normal pool.

It is the third defect this week with the same shape: a value that was correct
where it was chosen, carried unchanged into a regime where it is wrong, and
recorded as a fact about the world. tmux's `-e`, `compare()`'s denominator,
this. In all three the record of the failure said nothing — `Error: None`, a
mean over the wrong set, a task that simply never solves.

## The wall is the model's, and the other scaffold is standing at it too

Seven mechanisms here aimed at the same failure: the agent finishes, says it
verified its work, and the grader disagrees. Five of the seven failed for the
same reason — they ask the agent to judge the work with the judgement that
produced it. That was inferred from seven of my own failures, which is a weak
place to argue from.

claude-code, same model, same benchmark, settles it. Its failures end like
this:

    kv-store-grpc      3% of budget   "All done and verified working."
    query-optimize    20%             "Everything checks out."
    dna-insert        32%             "Done. The solution is written to ..."
    sam-cell-seg      90%             "Everything passes. The task is complete."
    filter-js         94%             "Done. `/app/filter.py` is complete and
                                       fully verified."

Every one scored zero. Counting the claim across whole runs:

                      failures claiming success    solves claiming success
    claude-code            12/23   (52%)                61/65   (94%)
    crux                   17/27   (63%)                56/60   (93%)
    crux, fixed            11/15   (73%)                53/56   (95%)

A claim of success is made by ~94% of runs that pass and by 52-73% of runs that
fail — a likelihood ratio between 1.3 and 1.8, for both agents. The signal is
nearly worthless, and it is *equally* worthless in the scaffold that is
currently ahead.

So this is not a gap between the two harnesses. It is a property of the model
at this size, and the best available scaffold does not clear it either. Seven
mechanisms is enough; an eighth aimed at the same thing would be measuring the
same wall.

What that leaves, and what the week's actual gains have all been: defects.
`</commands>`, tmux's `-e`, the 600-second ceiling — three faults that cost
real tasks and cost nothing at inference time to fix. None of them needed the
model to be better at anything.

### One reading I checked and dropped

Elapsed time per trial looked at first like an unfairness: claude-code trials
running to 202% and 269% of the nominal budget against crux being cut at 101%.
It is not. `finished_at - started_at` covers environment setup and the verifier
as well as the agent, and both agents overrun on the same tasks at similar
rates — 18% of claude-code's trials, 12% of crux's. The agent-execution phase
is the only clock that means anything here, and on that clock nobody is getting
extra.

## Being fair to the other scaffold, and getting it wrong in its favour

pi's headline 53.9% was not pi's fault. Sixteen of its 89 trials never ran:
each one downloads nvm from GitHub at startup, the download failed, and the
trial died under an exception named `NonZeroAgentExitCodeError` — which sounds
like the agent's and is not.

`Trial.category` decided "this is the box's fault" from a whitelist of
exception names, and a whitelist is never finished: 114 trials in this corpus
carrying endpoint 500s, setup timeouts and missing reward files were falling
through to `out_of_turns`, read as the agent running out of steps.

So it became a rule that does not depend on the name: **an exception plus no
agent activity at all is the environment's**.

The first version asked whether the trial had steps. That is a Terminus-shaped
question. pi writes `agent/pi` and `agent/pi.txt` and records no step count for
any of its 89 trials, so the rule read zero for all of them — and excused pi's
seven *genuine* agent timeouts along with its sixteen real setup deaths. I was
one command from reporting pi at 72.3% when the honest number is 65.8%.

The rule written to be fair to pi inflated it, by exactly the move this file
keeps recording: asking someone else's data a question shaped like my own.

Output tokens are the agent-independent version, and harbor records them for
every agent. On this corpus the split is exact:

    pi           16 NonZeroAgentExitCodeError   output tokens = 0   excluded
                  7 AgentTimeoutError           output tokens > 0   pi's loss
                 66 no exception                output tokens > 0
    claude-code   no trial with zero output tokens at all

### Three scaffolds, same model, same budget, paired

    claude-code vs pi                 73 tasks   71.2%  65.8%    8/12   p=0.50
    pi vs crux (ft-clean)             60         70.0%  75.0%   10/7    p=0.63
    pi vs crux (think-clean)          57         73.7%  77.2%    8/6    p=0.79
    claude-code vs crux (ft-clean)    75         73.3%  77.3%    9/6    p=0.61

crux > claude-code > pi, and not one pair separates. Against the earlier
record — "crux and pi tied, claude-code ahead of crux 10-5 on 71 tasks" — the
claude-code column has flipped after the parser, tmux and timeout fixes, to
9-6 the other way.

The pi rows rest on 57 and 60 usable tasks, because pi lost 16 to its own
installer and the crux arms lost several to stalls. At that size the
resolution is about ±12 points. The lead is a direction, not a result.

## Interleaved thinking, priced

The eighth mechanism is the only one that was ever measured on what it costs.
Input tokens per trial, median, over completed trials:

    base-crux       plain                782,391    output  55,205
    ft-clean        file tools         1,035,144    output  92,053
    base-cc         claude-code        1,686,917    output  51,373
    think-clean     interleaved        3,853,027    output 107,298

**4.9x the input of plain crux, and 2.3x claude-code's**, for a score of +2.50
against claude-code where file tools get +1.25 — the same patch of noise. Cache
rates are comparable on both sides (95% and 98%), so the uncached prefill is
about 5x as well; this is not an artifact of cache accounting.

Added to what was already measured — reasoning 48% longer, more steps not
fewer, context per step doubled — the mechanism does exactly what the model
card says it does, at five times the compute, for nothing.

It also explains a slowdown that looked like an endpoint fault. Single-stream
throughput fell from 127 tok/s to 9.2 while three arms ran. The GPUs were at
100% and the scheduler said why:

    Prefill batch, #new-seq: 1, #new-token: 8192, #pending-token: 154976,
    #running-req: 16, full token usage: 0.15

One 155,000-token context being chunk-prefilled nineteen chunks at a time, with
KV cache only 15% used: all prefill, no decode. The suspect was claude-code,
whose contexts are twice crux's. It was the interleaved arm, whose contexts are
five times.

**The lesson is about what gets measured.** Seven mechanisms were judged on
score alone, and score alone cannot distinguish "did nothing" from "did what it
promised, at five times the price". The second is a worse result and it reads
identically in a p-value.

## What the stall actually cost, counted

The 600-second ceiling was found in two Terminal-Bench arms and priced there:
eight trials, twelve slot-hours. Counting it across the whole corpus, by the
two strings it leaves behind — `failed to send non-blocking keys` and
`timeout value=600.0`:

    39 trials

    regex-chess           8      swe100b (SWE-bench)   10
    write-compressor      6      ft-clean               5
    dna-assembly          5      think-clean            4
    install-windows       4      ft6                    3
    circuit-fibsqrt       3      full-fixed             2
    adaptive-rejection    2      ...

**Ten of them are the SWE-bench run, and all ten are django.** Large repos and
long test runs make long model turns, which is exactly what a 600-second
per-call ceiling cuts. So `SWE-bench Verified 82.0%` — 73 of 89 — was computed
on a denominator that excluded ten trials killed by a constant of mine. The
rate may hold; the record should say what it rests on.

### It also finishes the Terminal-Bench story

Fourteen tasks have never been solved by crux in three or more attempts. Ten of
them are solved by nobody — claude-code and pi fail them too. Four are solved by
someone else:

    dna-assembly            pi          crux: stall
    install-windows-3.11    claude-code crux: stall, twice of three runs
    qemu-startup            both        crux: tmux -e, then timeout
    protein-assembly        both        crux: two honest zeros

So three of the four tasks where crux is specifically behind were harness
defects, and one is a real capability gap. That is the whole crux-specific
deficit against claude-code, and it was almost all mine.

`ft-fixed` is the first run to carry all three fixes, and `swe-stall` re-runs
the ten django trials that the ceiling took.

## Three readings of the same gap, and the third reverses the first two

The one structural difference that survived every other explanation was that
crux "grinds": more actions, more tokens, worse results on the hard tasks. It
went through three measurements.

**First**, trajectory steps: `extract-elf` 221 against 16, `mailman` 512
against 61 — up to 14x. That comparison was meaningless. A claude-code
trajectory step *is* a tool call (174 steps, 174 tool calls; medians 31 and 33),
while a crux step is a model turn carrying 1.85 shell commands. Two different
units.

**Second**, in comparable units — crux's real shell commands against
claude-code's tool calls, over the nine tasks crux times out on:

    gpt2-codegolf        103 :  10    10.3x
    make-mips-interp     430 :  65     6.6x
    extract-elf          122 :  19     6.4x
    feal-linear          267 :  84     3.2x
    extract-moves        325 : 174     1.9x
                                median 3.1x

Real, and much smaller than 14x.

**Third**, model turns — the axis that costs wall-clock and money:

                   median   total over 89
    crux              46         5,906
    claude-code       80        10,190

    crux uses more turns on 17 of 89 tasks

**claude-code takes 1.7x the model turns.** Input tokens per trial go the same
way: 1.09M for crux against 1.69M. So crux is not the inefficient one. It runs
fewer, denser turns; claude-code runs many more turns, most of them text, with
a heavily packed Bash call every third one.

The gap between the two — −1.18%, p=1.000 — is not explained by grinding,
because there is no grinding to explain it with. Three measurements of one
claim, and the third contradicts the first two.

The lesson is one this file already records and I repeated anyway: **a process
metric compared across two agents means nothing until you have checked what one
unit is on each side.** "Steps" was a word that meant different things.
