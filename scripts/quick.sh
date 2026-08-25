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
# The four that have never been solved across ten runs, plus two that fail
# close to the line. Optimizing against tasks already solved teaches nothing.
CONTESTED="${CONTESTED:-qemu-alpine-ssh torch-pipeline-parallelism filter-js-from-html gpt2-codegolf extract-elf kv-store-grpc}"

if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
# Free model by default: iterating on a paid one is how ~470 CNY went on runs
# that mostly re-confirmed known failures.
MODEL="${MODEL:-openrouter/stealth/ox-alpha}"
VARIANT="${VARIANT:-default}"
AGENT="${AGENT:-crux.agent:CruxAgent}"
N_CONCURRENT="${N_CONCURRENT:-10}"

INCLUDE=""
for t in ${CANARIES} ${CONTESTED}; do
  INCLUDE="${INCLUDE} --include-task-name terminal-bench/${t}"
done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages | head -1)
export PYTHONPATH="$(pwd)/src:${SP}"

echo "agent   : ${AGENT}  variant=${VARIANT}${AK_EXTRA:+  extra=${AK_EXTRA}}"
echo "model   : ${MODEL}"
echo "canary  : ${CANARIES}"
echo "contest : ${CONTESTED}"
echo

AK=""
case "${AGENT}" in
  crux.agent:CruxAgent) AK="--ak variant=${VARIANT}" ;;
esac
# Any further agent kwargs, space separated, e.g. AK_EXTRA="reasoning_effort=low".
# Kept out of VARIANT because a variant is a named, reproducible configuration
# and a one-off A/B is not; mixing the two makes past scores unreadable.
for kv in ${AK_EXTRA:-}; do
  AK="${AK} --ak ${kv}"
done

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
