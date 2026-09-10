#!/bin/bash
# Run one arm over the tasks a change actually aims at, plus a regression guard.
#
# Why not the full 89: a change aimed at ten tasks is *diluted* by the other
# seventy-nine, which contribute nothing but noise to a paired test.
#
# And why only one arm. Every target below scores 0 across the three full arms
# run so far -- 0/9, task by task -- so the question is not "does this arm beat
# that one at some rate", it is "does a task that has never been solved get
# solved". That is binary, the historical control is unambiguous, and it does
# not need a concurrent twin. Dropping the control arm is what makes this fast:
# the targets are the *slowest* tasks on the board, so two arms over them is no
# cheaper in wall clock than a full run.
#
# The guard set is the exception and it does need care: those tasks pass today,
# so a rate really is being compared, and a slow arm can fail them for reasons
# that have nothing to do with the change. Read a guard failure as a question,
# not a verdict, and confirm it on the full set before believing it.
#
# Usage:  targeted.sh <arm-name> <attempts> [extra --ak args...]
set -euo pipefail

ARM="${1:?arm name}"
ATTEMPTS="${2:-3}"
shift 2 || true

# --- the targets -------------------------------------------------------------
# Ten tasks that carry the measured failure: two or more consecutive no-action
# turns of 20,000+ thinking characters. 50 of 441 trials show it and 41 of those
# failed, against a base rate near 45%.
THINKING_DEATH=(
  schemelike-metacircular-eval circuit-fibsqrt feal-differential-cryptanalysis
  adaptive-rejection-sampler polyglot-rust-c winning-avg-corewars
  path-tracing-reverse regex-chess torch-pipeline-parallelism
  extract-moves-from-video
)
# Three tasks where pi was killed with the container, not by its own logic:
# seven exit-137 trials, from `-j$(nproc)` reading the host's 192 cores and from
# the agent's own `pkill -f` matching its shell.
KILLED=(rstan-to-pystan install-windows-3.11 mcmc-sampling-stan)
# One task that died before taking an action because its instruction begins with
# a hyphen and nothing passed pi an option terminator.
PARSE=(pytorch-model-recovery)

# --- the guard ---------------------------------------------------------------
# Fast tasks that pass 2/2 today. They are here to fail: a change that helps the
# targets and breaks these is not an improvement, and a targeted run cannot see
# that on its own. Chosen by runtime so the guard is nearly free.
GUARD=(
  log-summary-date-ranges modernize-scientific-stack git-leak-recovery
  kv-store-grpc nginx-request-logging fix-git
)

ARGS=()
for t in "${THINKING_DEATH[@]}" "${KILLED[@]}" "${PARSE[@]}" "${GUARD[@]}"; do
  ARGS+=(--include-task-name "terminal-bench/$t")
done

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=/scratch/crux-next-src:/root/.local/share/uv/tools/harbor/lib/python3.13/site-packages
cd /scratch/crux

exec harbor run --dataset terminal-bench/terminal-bench-2-1 \
  --agent crux.pi_agent:CruxPiAgent --model openai/qwen3.8-27b \
  --n-attempts "$ATTEMPTS" --n-concurrent 10 --jobs-dir "/scratch/$ARM" \
  --env docker --yes \
  --ak variant=default --ak model_api=openai-completions \
  --ak bundle=/scratch/crux/pi-nvm-budget.tar.gz \
  --ak sections=doing_full,cc_tools \
  "$@" \
  --agent-timeout-multiplier 8.0 --env-file /scratch/crux/.env \
  "${ARGS[@]}"
