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

DATASET="${DATASET:-terminal-bench/terminal-bench@latest}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
N_TASKS="${N_TASKS:-}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-24}"

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="$(pwd)/src:${SP}"

echo "dataset : ${DATASET}"
echo "model   : ${MODEL}"
echo "jobs    : ${JOBS_DIR}"
echo "concur  : ${N_CONCURRENT}   attempts: ${N_ATTEMPTS}"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent-import-path "crux.agent:CruxAgent" \
  --model "${MODEL}" \
  ${N_TASKS:+--n-tasks "${N_TASKS}"} \
  --n-attempts "${N_ATTEMPTS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env-file .env \
  --env docker \
  --yes
