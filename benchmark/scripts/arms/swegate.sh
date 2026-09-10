#!/usr/bin/env bash
# The 16 tasks SWE-bench Verified failed, re-run with the submit gate on.
#
# Paired, and on exactly the set the gate is aimed at. Of those 16 failures:
#
#   15 stopped on a green light with more than half the budget unspent
#   15 missed by two tests or fewer
#    0 were timeouts
#
# The extreme case is pydata__xarray-4687: 1717 of 1718 tests passed, the agent
# declared itself done on its own check, and stopped having used 3.2% of a
# 400-minute budget.
#
# swe100b ran with the gate off. This is the same tasks with confirm_gate=1, so
# `crux submit` refuses once and asks for one more check bound from the task's
# own words. Everything else is identical.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
ARGS=""
while read -r t; do [ -n "$t" ] && ARGS="$ARGS -t $t"; done < /tmp/swe_failed.txt
exec python3 -m crux.cli bench \
  --dataset swe-bench/swe-bench-verified \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 8 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/swe-gate \
  --env-file /scratch/crux/.env \
  --agent-kwarg confirm_gate=1 \
  $ARGS
