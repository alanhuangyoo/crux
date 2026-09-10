#!/usr/bin/env bash
# The 29 tasks the crux baseline failed.
#
# NOT a gate comparison, despite the name and despite what this comment used to
# claim. It leaves edit_debt_limit unset and so does full_run.sh, so both take
# the default of 12 and both have the gate. The two differ in their task list
# and their concurrency and in nothing else.
#
# Which means their 16 shared tasks measured run-to-run noise and were read as
# the gate working: 9 versus 5, with four disagreements all falling one way.
# Four one-directional flips out of sixteen is p=0.125 under a coin, and a
# 29-task run resolves about +/-15 points, so that reading never had the
# resolution it was given.
#
# gateoff.sh is the control this needed. Keep this file for the arm it actually
# is: the hard subset, everything on.
cd /scratch/crux
export PATH="$HOME/.local/bin:$PATH"
SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="/scratch/crux/src:${SP}"
exec python3 -m crux.cli bench \
  --dataset terminal-bench/terminal-bench-2-1 \
  --model openai/qwen3.8-27b \
  --attempts 1 --concurrent 8 --allow-concurrent-jobs \
  --agent-timeout-multiplier 8 \
  --jobs-dir /scratch/gate-on \
  --env-file /scratch/crux/.env \
  -t adaptive-rejection-sampler -t build-pov-ray -t chess-best-move -t db-wal-recovery -t dna-assembly -t dna-insert -t extract-elf -t extract-moves-from-video -t feal-linear-cryptanalysis -t filter-js-from-html -t gcode-to-text -t gpt2-codegolf -t install-windows-3.11 -t mailman -t make-doom-for-mips -t make-mips-interpreter -t mcmc-sampling-stan -t model-extraction-relu-logits -t mteb-retrieve -t path-tracing -t path-tracing-reverse -t protein-assembly -t qemu-startup -t raman-fitting -t regex-chess -t sam-cell-seg -t schemelike-metacircular-eval -t torch-pipeline-parallelism -t video-processing
