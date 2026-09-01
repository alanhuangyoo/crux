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
# The full run is the confirmation step, so it keeps the paid model. Anything
# being developed should go through scripts/quick.sh on the free one first.
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
N_TASKS="${N_TASKS:-}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-24}"
VARIANT="${VARIANT:-default}"

# These four declare gpus=1. Docker here has no nvidia runtime, and wiring one
# up would contend with the training job that owns all eight cards. Excluding
# them is not cosmetic: the validation error propagates and aborts the whole
# job, so leaving them in cancels every other trial in flight (docs/ENVIRONMENT.md).
#
# The names are qualified by dataset org, so a hardcoded "terminal-bench/" prefix
# silently matches nothing on any other dataset -- it looked like the exclusion
# was active through 88 trials of a terminal-bench-pro run where it was a no-op.
# Pro happens to declare gpus=0 on all 200 tasks so nothing was at risk, but the
# next dataset may not. Derive the org, and leave the list empty when it does not
# apply rather than pretending to exclude.
GPU_TASKS="${GPU_TASKS:-exam-pdf-eval fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader}"
ORG="${DATASET%%/*}"
EXCLUDE=""
if [ "${ORG}" = "terminal-bench" ]; then
  for t in ${GPU_TASKS}; do EXCLUDE="${EXCLUDE} --exclude-task-name ${ORG}/${t}"; done
else
  GPU_TASKS="(none -- ${ORG} tasks declare no GPU requirement)"
fi

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="$(pwd)/src:${SP}"

AGENT="${AGENT:-crux.agent:CruxAgent}"
AK=""
case "${AGENT}" in
  crux.*) AK="--ak variant=${VARIANT}" ;;
esac
for kv in ${AK_EXTRA:-}; do AK="${AK} --ak ${kv}"; done

echo "agent   : ${AGENT}"
echo "dataset : ${DATASET}"
echo "model   : ${MODEL}"
echo "jobs    : ${JOBS_DIR}"
echo "variant : ${VARIANT}"
echo "concur  : ${N_CONCURRENT}   attempts: ${N_ATTEMPTS}"
echo "excluded: ${GPU_TASKS} (need GPU)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent "${AGENT}" \
  ${AK} \
  --model "${MODEL}" \
  ${N_TASKS:+--n-tasks "${N_TASKS}"} \
  --n-attempts "${N_ATTEMPTS}" \
  ${EXCLUDE} \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env-file .env \
  --env docker \
  --yes
