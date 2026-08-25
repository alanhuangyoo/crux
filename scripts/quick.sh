#!/usr/bin/env bash
# Fast iteration set — a fixed slice, not the whole benchmark.
#
# A full run is 85 tasks and hours of wall clock, which is far too slow a loop
# to develop against and wasteful to repeat. This set is chosen so a change
# shows up quickly and in both directions:
#
#   canaries   tasks solved consistently. If one stops passing, the change
#              broke something, and that matters more than any gain.
#   contested  tasks that failed close to the line. These are where a real
#              improvement shows up first.
#
# Roughly 10 tasks, ~15 minutes, a couple of dollars. Confirm anything that
# looks like a win on the full set before believing it.
set -euo pipefail

CANARIES="${CANARIES:-log-summary-date-ranges openssl-selfsigned-cert pypi-server regex-log}"
CONTESTED="${CONTESTED:-kv-store-grpc extract-elf torch-tensor-parallelism dna-assembly filter-js-from-html build-pov-ray}"

if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
VARIANT="${VARIANT:-default}"
AGENT="${AGENT:-crux.agent:CruxAgent}"
N_CONCURRENT="${N_CONCURRENT:-10}"

INCLUDE=""
for t in ${CANARIES} ${CONTESTED}; do
  INCLUDE="${INCLUDE} --include-task-name terminal-bench/${t}"
done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="$(pwd)/src:${SP}"

echo "agent   : ${AGENT}  variant=${VARIANT}"
echo "model   : ${MODEL}"
echo "canary  : ${CANARIES}"
echo "contest : ${CONTESTED}"
echo

AK=""
case "${AGENT}" in
  crux.agent:CruxAgent) AK="--ak variant=${VARIANT}" ;;
esac

harbor run \
  --dataset "${DATASET}" \
  --agent "${AGENT}" \
  ${AK} \
  --model "${MODEL}" \
  ${INCLUDE} \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env-file .env \
  --env docker \
  --yes
