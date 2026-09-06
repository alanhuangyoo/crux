#!/usr/bin/env bash
# Third arm on the same 29 tasks the other two run: the ones the crux baseline
# scored 0 on. Same tasks, so the three are directly comparable, and a subset
# where the baseline is 0 cannot lose to noise in the downward direction.
#
#   full-fixed  no gate
#   gate-on     verification-debt gate (blocks the 13th unchecked edit)
#   submit-gate `crux submit` refuses once and asks for one more check bound
#               from the task's own words
#
# The third is the one the budget measurement supports: failures stop
# voluntarily with two thirds of their budget unspent, on a green light the
# agent wrote itself.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"

# wait for gate-on's own redo to finish too, so the box is not oversubscribed
while ps -eo args | grep -q -- "[h]arbor run.*jobs-dir /scratch/gate-on"; do sleep 120; done
echo "$(date +%H:%M) gate-on 结束,起 submit-gate"

ARGS=""
while read -r t; do [ -n "$t" ] && ARGS="$ARGS -t $t"; done < /tmp/gate_tasks.txt
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 8 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/submit-gate \
  --env-file /scratch/crux/.env \
  --agent-kwarg confirm_gate=1 \
  --agent-kwarg edit_debt_limit=0 \
  $ARGS
