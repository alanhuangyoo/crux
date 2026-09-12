#!/bin/bash
# One arm, alone on the box, for an absolute number rather than a comparison.
#
# Every A/B here runs two arms of 8 concurrent trials, so the engine carries 16
# streams. A single arm carries 8. The trial-level concurrency is the same 8
# either way -- and 8 is the level this benchmark needs, worth +9 points over 24
# because the 900s median budget makes timeouts the binding constraint -- but the
# endpoint is half as loaded, responses come back faster, and fewer trials run
# out of time. So a number from this script is NOT comparable with an arm from
# ab.sh; it is comparable with a leaderboard submission, which is also one run
# alone at 8.
#
# Usage: one.sh <name> <bundle> [extra --ak args...]
set -euo pipefail
NAME="$1"; BUNDLE="$2"; shift 2
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/scratch/crux-next-src:/root/.local/share/uv/tools/harbor/lib/python3.13/site-packages

run() {
  cd /scratch/crux
  harbor run --dataset terminal-bench/terminal-bench-2-1 \
    --agent crux.pi_agent:CruxPiAgent --model openai/Qwen3.8-Flash-Next-FP8 \
    --n-attempts 1 --n-concurrent 8 --jobs-dir "/scratch/$1" --env docker --yes \
    --ak variant=default --ak model_api=openai-completions \
    --ak bundle="$2" --ak sections=doing_full,cc_tools \
    --ak pi_env=PI_TIME_BUDGET_SEC=7200,PI_STOP_BUDGET_SHARE=0.8,PI_MAX_OUTPUT_TOKENS=8192,PI_ESCALATE_SPENT_CEILING=1 \
    --agent-timeout-multiplier 8.0 --env-file /scratch/crux/.env "${@:3}" \
    --exclude-task-name terminal-bench/exam-pdf-eval \
    --exclude-task-name terminal-bench/fp8-rmsnorm-gemm \
    --exclude-task-name terminal-bench/jax-speedrun-gpu \
    --exclude-task-name terminal-bench/math-eval-grader
}

rm -rf "/scratch/${NAME}"
setsid nohup bash -c "$(declare -f run); run ${NAME} ${BUNDLE} $*" > "/scratch/${NAME}.log" 2>&1 </dev/null &
echo "arm ${NAME} (${BUNDLE##*/}) pid=$!"
