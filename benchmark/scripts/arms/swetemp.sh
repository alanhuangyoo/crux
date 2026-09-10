#!/usr/bin/env bash
# The same 16 SWE-bench tasks a third time, at a lower sampling temperature.
#
# crux has never set one. Terminus only passes a temperature when it is
# explicitly configured and crux never configures it, so every number this
# project has produced was sampled at the server default -- the
# maximum-variance setting. Measured on the endpoint: the same prompt five
# times gives three different answers with no temperature set, and one answer
# at temperature 0.
#
# That is the likeliest explanation of everything variance-shaped here:
#
#   60%     of SWE-bench failures solve on a plain re-run, nothing changed
#   15-16%  of tasks flip between two runs of one configuration
#   +/-8    points is all an 89-task run can resolve
#
# and of why both gates looked effective: in a system this noisy, anything
# selected on failure looks better when re-run.
#
# 0.2 rather than 0: Qwen3's own card warns that greedy decoding in thinking
# mode can fall into repetition loops, and this is a thinking model.
#
# Comparable to swe-gate (8/14) and swe-nogate (9/15), which are the same tasks
# at the default temperature with the submit gate on and off.
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
  --jobs-dir /scratch/swe-temp02 \
  --env-file /scratch/crux/.env \
  --agent-kwarg temperature=0.2 \
  $ARGS
