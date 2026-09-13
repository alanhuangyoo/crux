#!/bin/bash
# One arm, alone on the box, for an absolute number rather than a comparison.
#
# Concurrency is 16, not the 8 this project used for a year. 8 was derived when
# the 900s median budget made timeouts the binding constraint and 24 lost 9
# points to it. Measured again after the hangs were fixed: with 8 trials in
# flight the engine was serving **1 to 2 requests at a time** and 2% of its KV
# pool, because a trial spends most of its life in bash rather than waiting on
# the model. A mixture-of-experts model batched one request at a time is the
# worst case for it, and the run took six hours.
#
# The same shape as `--context-length 32768`: a number that was right where it
# was chosen, carried into a regime where it is not, and read as a property of
# the box.
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
# SECTIONS overrides which prompt sections are appended; see benchmark/src/crux/prompts.py
export SECTIONS="${SECTIONS:-doing_full,cc_tools}"
NAME="$1"; BUNDLE="$2"; shift 2
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/scratch/crux-next-src:/root/.local/share/uv/tools/harbor/lib/python3.13/site-packages

# PI_MAX_OUTPUT_TOKENS must stay *below* the model's own maxTokens, which
# pi_agent derives as min(32768, window // 4) -- 32768 at the 262144 this endpoint
# now serves. 16384 leaves the escalation a step to take. Setting the two equal is
# a silent no-op: the truncation recovery it exists for can never raise anything,
# which cost a full round once already.
run() {
  cd /scratch/crux
  harbor run --dataset terminal-bench/terminal-bench-2-1 \
    --agent crux.pi_agent:CruxPiAgent --model openai/Qwen3.8-Flash-Next-FP8 \
    --n-attempts 1 --n-concurrent 16 --jobs-dir "/scratch/$1" --env docker --yes \
    --ak variant=default --ak model_api=openai-completions \
    --ak bundle="$2" --ak sections="${SECTIONS:-doing_full,cc_tools}" \
    --ak pi_env=PI_TIME_BUDGET_SEC=7200,PI_STOP_BUDGET_SHARE=0.8,PI_MAX_OUTPUT_TOKENS=16384,PI_ESCALATE_SPENT_CEILING=1 \
    --agent-timeout-multiplier 8.0 --env-file /scratch/crux/.env "${@:3}" \
    --exclude-task-name terminal-bench/exam-pdf-eval \
    --exclude-task-name terminal-bench/fp8-rmsnorm-gemm \
    --exclude-task-name terminal-bench/jax-speedrun-gpu \
    --exclude-task-name terminal-bench/math-eval-grader
}

rm -rf "/scratch/${NAME}"
setsid nohup bash -c "export SECTIONS=$SECTIONS; $(declare -f run); run ${NAME} ${BUNDLE} $*" > "/scratch/${NAME}.log" 2>&1 </dev/null &
echo "arm ${NAME} (${BUNDLE##*/}) pid=$!"
