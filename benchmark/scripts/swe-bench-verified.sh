#!/usr/bin/env bash
# SWE-bench Verified, 100 tasks sampled at random (seed 20260906) from all 500.
# The first cut of this run took the alphabetically first 100, which is 22
# astropy + 78 django -- two repos out of twelve. A number from that sample is
# not comparable to any published SWE-bench Verified result, so it was thrown
# away and re-drawn. The seed is in /tmp/sample.py; the draw is reproducible.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
ARGS=""
while read -r t; do ARGS="$ARGS -t $t"; done < /tmp/swe_sample.txt
exec python3 -m crux.cli bench \
  --dataset swe-bench/swe-bench-verified \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 10 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/swe100b \
  --env-file /scratch/crux/.env \
  $ARGS
