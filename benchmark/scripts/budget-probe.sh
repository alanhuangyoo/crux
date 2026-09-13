#!/bin/bash
# Is it time, or is it capability?
#
# After the compaction and hang fixes, 17 of v6's 28 failures ended with the
# clock: 7 killed by the harness having worked for 117 of their 121 minutes, and
# 10 more ending on `toolUse` -- still issuing tool calls -- when the agent's own
# 7200s budget ran out. None of them was looping: across 886 tool calls
# `make-doom-for-mips` repeated 1.8% of them, and the last quarter of its run
# repeated 2.7% of what came earlier. They were working and ran out.
#
# This run gives exactly those tasks twice the budget and nothing else. Their
# baseline needs no statistics: all 17 scored zero, so any solve is the answer,
# and a 0/17 does not need a noise floor either.
#
# Usage: budget-probe.sh <name> <bundle> "<task> <task> ..."
set -euo pipefail
NAME="$1"; BUNDLE="$2"; TASKS="$3"; shift 3
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/scratch/crux-next-src:/root/.local/share/uv/tools/harbor/lib/python3.13/site-packages

INCLUDES=""
for t in $TASKS; do INCLUDES="$INCLUDES --include-task-name terminal-bench/$t"; done

# PI_MAX_OUTPUT_TOKENS must stay *below* the model's own maxTokens, which
# pi_agent derives as min(32768, window // 4) -- 32768 at the 262144 this endpoint
# now serves. 16384 leaves the escalation a step to take. Setting the two equal is
# a silent no-op: the truncation recovery it exists for can never raise anything,
# which cost a full round once already.
run() {
  cd /scratch/crux
  harbor run --dataset terminal-bench/terminal-bench-2-1 \
    --agent crux.pi_agent:CruxPiAgent --model openai/Qwen3.8-Flash-Next-FP8 \
    --n-attempts 1 --n-concurrent 8 --jobs-dir "/scratch/$1" --env docker --yes \
    --ak variant=default --ak model_api=openai-completions \
    --ak bundle="$2" --ak sections=doing_full,cc_tools \
    --ak pi_env=PI_TIME_BUDGET_SEC=14400,PI_STOP_BUDGET_SHARE=0.8,PI_MAX_OUTPUT_TOKENS=16384,PI_ESCALATE_SPENT_CEILING=1 \
    --agent-timeout-multiplier 16.0 --env-file /scratch/crux/.env "${@:3}"
}

rm -rf "/scratch/${NAME}"
setsid nohup bash -c "$(declare -f run); run ${NAME} ${BUNDLE} ${INCLUDES} $*" > "/scratch/${NAME}.log" 2>&1 </dev/null &
echo "budget probe ${NAME} (${BUNDLE##*/}) pid=$!"
echo "tasks: $(echo $TASKS | wc -w), budget 14400s, timeout multiplier 16"
