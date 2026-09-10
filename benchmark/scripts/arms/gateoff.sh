#!/usr/bin/env bash
# The control arm that was missing.
#
# `gate-on` and `full-fixed` were set up as gate-on/gate-off and are not: both
# scripts leave edit_debt_limit unset, so both take the default of 12 and both
# have the gate. They differ in their task list and nothing else, which makes
# their 16 shared tasks a measurement of run-to-run noise (9 vs 5, four
# disagreements all one way) rather than of the gate.
#
# This is the same 29 tasks with the gate actually off. Three arms then differ
# by one thing each:
#
#   gate-on      edit-debt gate on   (default 12)
#   gate-off     edit-debt gate off  (this)
#   submit-gate  edit-debt off, `crux submit` demands a second pass
#
# Worth running on this subset in particular: the gate fired in 8 of 19 trials
# here and in 0 of 68 across the full 89, so the hard tasks are the only place
# it has anything to act on.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
ARGS=""
while read -r t; do [ -n "$t" ] && ARGS="$ARGS -t $t"; done < /tmp/gate_tasks.txt
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 8 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/gate-off \
  --env-file /scratch/crux/.env \
  --agent-kwarg edit_debt_limit=0 \
  $ARGS
