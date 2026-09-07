# The arms, and one that was not an arm

Every file here is exactly what ran. They are kept because the mistake below is
easier to make than to notice.

## The mistake

`gate_test.sh` was written as the gate-on arm and `full_run.sh` as the gate-off
one. Neither passes `edit_debt_limit`, so both take the default of 12 and
**both have the gate**. They differ in their task list and their concurrency.

Their 16 shared tasks were then read as evidence the gate worked -- 9 versus 5,
four disagreements all one way -- when what they measured was two runs of one
configuration. Four one-directional flips out of sixteen is p=0.125 under a
coin, and a 29-task run resolves about +/-15 points.

What gives it away is not the config, which looks the same either way, but the
trajectories:

    gate fired in   8 of 19 trials   gate-on   (the 29 hard tasks)
                    0 of 68 trials   full-fixed (the full 89)
                    1 of 99 trials   swe100b

Both arms could fire it. Only the hard subset ever does, because the parser
fixes changed the editing behaviour enough that debt rarely reaches 12 -- which
is itself the more interesting finding, and was hidden by the mislabelling.

## The arms

| script | task set | edit-debt gate | submit gate |
|---|---|---|---|
| `full_run.sh` | all 89 | on (default 12) | off |
| `gate_test.sh` | the 29 the baseline failed | on (default 12) | off |
| `gateoff.sh` | the same 29 | **off** | off |
| `submitgate.sh` | the same 29 | off | **on** |

The last three differ from each other by one thing each, which is the only
arrangement that lets a flip be attributed.

## Reading them

    crux watch --compare /scratch/gate-on /scratch/gate-off --baseline /scratch/base-crux

Paired on shared tasks with a sign test. Never total against total: the arms do
not finish the same tasks at the same time, and a partly finished run reads
high because failed trials take about twice as long as solved ones.


## The full set

| script | benchmark | task set | edit-debt gate | submit gate | temperature |
|---|---|---|---|---|---|
| `full_run.sh` | TB 2.1 | all 89 | on (12) | off | server default |
| `gate_test.sh` | TB 2.1 | the 29 the baseline failed | on (12) | off | server default |
| `gateoff.sh` | TB 2.1 | the same 29 | **off** | off | server default |
| `submitgate.sh` | TB 2.1 | the same 29 | off | **on** | server default |
| `allon.sh` | TB 2.1 | all 89 | on (12) | **on** | server default |
| `allon_redo.sh` | TB 2.1 | what `allon` lost to a loaded box | on (12) | on | server default |
| `swegate.sh` | SWE-bench | the 16 it failed | on (12) | **on** | server default |
| `swenogate.sh` | SWE-bench | the same 16 | on (12) | off | server default |
| `swetemp.sh` | SWE-bench | the same 16 | on (12) | off | **0.2** |

## Two operational mistakes these scripts carry

**A guard that could not match.** `allon_redo.sh` waits for the main run to
finish before re-running what it lost. Its first version was

    while ps -eo args | grep -q -- "[h]arbor run.*jobs-dir /scratch/all-on\$"

inside a quoted heredoc, so `\$` reached grep as an escaped dollar -- which
matches a literal `$` and therefore nothing. The guard fired immediately.

**Too many arms at once.** `all-on` lost 24 of 89 trials to
`Agent setup timed out after 360` and `Environment start timed out after 600`.
Zero of those appear in the four runs before it. What was different is that six
other arms were running alongside and the load average peaked at 153; setup
pulls an image, starts a container, opens tmux and uploads two files, and at
that load 360 seconds is not enough. The tmux probe crux adds to setup costs
94ms, measured, so it is not that.

The endpoint is the real constraint and it is fixed: four GPUs at 100%, 62
tok/s single-stream against 300 idle with 45 containers, 120 tok/s with 24.
Parallelism is free in coverage and not in the wall-clock of any one arm.
