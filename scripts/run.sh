#!/usr/bin/env bash
# Run the Crux agent against the benchmark.
#
# Concurrency is deliberately a fixed default rather than a per-run choice:
# tasks carry timeouts, so a different concurrency level shifts scores and
# makes runs incomparable. Keep it constant from tuning through submission.
set -euo pipefail

if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

# 2.1 is where the published leaderboard numbers live; @latest is Terminal-Bench 3,
# the frontier set, which is deliberately harder and not comparable to them.
DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
N_TASKS="${N_TASKS:-}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-24}"
VARIANT="${VARIANT:-default}"

# These four declare gpus=1. Docker here has no nvidia runtime, and wiring one
# up would contend with the training job that owns all eight cards. Excluding
# them is not cosmetic: the validation error propagates and aborts the whole
# job, so leaving them in cancels every other trial in flight (docs/ENVIRONMENT.md).
GPU_TASKS="${GPU_TASKS:-exam-pdf-eval fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader}"
EXCLUDE=""
for t in ${GPU_TASKS}; do EXCLUDE="${EXCLUDE} --exclude-task-name terminal-bench/${t}"; done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="$(pwd)/src:${SP}"

echo "dataset : ${DATASET}"
echo "model   : ${MODEL}"
echo "jobs    : ${JOBS_DIR}"
echo "variant : ${VARIANT}"
echo "concur  : ${N_CONCURRENT}   attempts: ${N_ATTEMPTS}"
echo "excluded: ${GPU_TASKS} (need GPU)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent "crux.agent:CruxAgent" \
  --ak "variant=${VARIANT}" \
  --model "${MODEL}" \
  ${N_TASKS:+--n-tasks "${N_TASKS}"} \
  --n-attempts "${N_ATTEMPTS}" \
  ${EXCLUDE} \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env-file .env \
  --env docker \
  --yes
