#!/usr/bin/env bash
# The run the question actually turns on: all 89, everything on.
#
# What "everything" means here, and why each piece is in:
#
#   edit-debt gate (default 12)  blocks the 13th edit made without running
#                                anything that checks it
#   submit gate    confirm_gate  `crux submit` refuses once and asks for one
#                                more check bound from the task's own words
#   bind-time probe              a check is run when it is bound, and told
#                                whether it already passes -- 88% of trials
#                                never see a bound check fail, including the
#                                ones that solved the task
#   budget notice                at "are you sure you are done", how much of
#                                the run is left. Failures stop voluntarily
#                                with two thirds of their budget unspent, and
#                                the agent has never been able to see that
#
# Read against base-cc paired on shared tasks, never total against total, and
# with the resolution in mind: two runs of one configuration disagree on
# 15-16% of tasks, so an 89-task run resolves about ±8 points.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 12 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/all-on \
  --env-file /scratch/crux/.env \
  --agent-kwarg confirm_gate=1
