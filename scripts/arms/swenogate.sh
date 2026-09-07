#!/usr/bin/env bash
# The control for swe-gate, and the reason its 5-of-10 cannot be read yet.
#
# Those 16 tasks were selected because they failed. Re-running anything
# selected on failure recovers some of it by regression to the mean alone --
# this benchmark's own run-to-run flip rate is 15-16% -- so "5 of 10 flipped
# with the gate on" is not yet a statement about the gate.
#
# Same 16 tasks, same everything, gate off. Whatever this arm recovers is what
# a re-run recovers; the difference is what the gate is worth.
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
  --jobs-dir /scratch/swe-nogate \
  --env-file /scratch/crux/.env \
  $ARGS
