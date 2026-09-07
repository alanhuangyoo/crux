#!/usr/bin/env bash
# The tasks all-on lost to a loaded box, re-run on a quiet one.
#
# 24 of 89 errored, and none of it was the configuration:
#
#   7  Agent setup timed out after 360
#   5  Environment start timed out after 600
#   2  Failed to start tmux session
#
# Zero of these appear in base-crux, full-fixed, gate-on or swe100b. What was
# different is that all-on ran with six other arms alongside it and the load
# average peaked at 153. Setup pulls an image, starts a container, opens tmux
# and uploads two files, all of it disk and CPU bound, and at that load 360
# seconds is not enough. The tmux probe this project adds to setup costs 94ms,
# measured, so it is not that.
#
# Same configuration -- nothing touching the agent has changed since all-on
# started -- so these merge into the same arm.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
# `\$` inside a quoted heredoc reaches grep as an escaped dollar, which matches
# a literal `$` and therefore nothing -- this guard fired immediately the first
# time it ran. The anchor has to be a real end-of-line, so the pattern is built
# where the shell can see it.
END='--jobs-dir /scratch/all-on$'
while ps -eo args | grep -q -- "[h]arbor run" && ps -eo args | grep -qE "$END"; do sleep 60; done
python3 - > /tmp/allon_left.txt <<'PY'
import sys
sys.path.insert(0, "/scratch/crux/src")
from crux.watch import unscored
for t in unscored("/scratch/all-on"):
    print(t)
PY
n=$(grep -c . /tmp/allon_left.txt)
echo "$(date +%H:%M) all-on 主跑结束,补 $n 道"
[ "$n" -eq 0 ] && exit 0
ARGS=""
while read -r t; do [ -n "$t" ] && ARGS="$ARGS -t $t"; done < /tmp/allon_left.txt
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 10 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/all-on-redo \
  --env-file /scratch/crux/.env \
  --agent-kwarg confirm_gate=1 \
  $ARGS
