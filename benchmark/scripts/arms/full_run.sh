#!/usr/bin/env bash
# The full 89 with every fix in place. Everything above 69.0% has been a
# projection from 13 tasks; this is the number that replaces it.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 12 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/full-fixed \
  --env-file /scratch/crux/.env
