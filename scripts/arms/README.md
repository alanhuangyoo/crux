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
