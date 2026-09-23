#!/bin/bash
# One change, two arms, launched together.
#
# Measured on this box: two arms of an identical configuration run *at the same
# time* differ by 3.8 points and flip 2 of 52 tasks. The same configuration run
# one after the other differed by 16.7 points. That gap is not randomness in the
# agent, it is the box changing between windows -- load from other tenants, the
# endpoint's state, throughput. So arms are never compared across time here.
#
# 3.8 points of noise on 89 tasks is the resolution. An effect smaller than that
# cannot be seen this way and needs a process measure instead -- truncation
# rate, actions per turn, tokens removed per compaction -- which are counted
# over thousands of turns rather than 89 outcomes.
#
# Usage: ab.sh <name> <bundle-A> <bundle-B> [extra --ak args...]
set -euo pipefail
# MODEL is the name the endpoint serves; override for a different deployment.
export MODEL="${MODEL:-Qwen3.8-27B-FP8}"
NAME="$1"; A="$2"; B="$3"; shift 3
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
    --agent crux.pi_agent:CruxPiAgent --model "openai/${MODEL:-Qwen3.8-27B-FP8}" \
    --n-attempts 1 --n-concurrent 8 --jobs-dir "/scratch/$1" --env docker --yes \
    --ak variant=default --ak model_api=openai-completions \
    --ak bundle="$2" --ak sections=doing_full,cc_tools,notes \
    --ak pi_env=PI_TIME_BUDGET_SEC=14400,PI_STOP_BUDGET_SHARE=0.8,PI_MAX_OUTPUT_TOKENS=16384,PI_ESCALATE_SPENT_CEILING=1 \
    --agent-timeout-multiplier 8.0 --env-file /scratch/crux/.env "${@:3}" \
    --exclude-task-name terminal-bench/exam-pdf-eval \
    --exclude-task-name terminal-bench/fp8-rmsnorm-gemm \
    --exclude-task-name terminal-bench/jax-speedrun-gpu \
    --exclude-task-name terminal-bench/math-eval-grader
}

rm -rf "/scratch/${NAME}-a" "/scratch/${NAME}-b"
setsid nohup bash -c "$(declare -f run); run ${NAME}-a $A $*" > "/scratch/${NAME}-a.log" 2>&1 </dev/null &
echo "arm A (${A##*/}) pid=$!"
setsid nohup bash -c "$(declare -f run); run ${NAME}-b $B $*" > "/scratch/${NAME}-b.log" 2>&1 </dev/null &
echo "arm B (${B##*/}) pid=$!"
