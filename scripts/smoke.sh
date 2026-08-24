#!/usr/bin/env bash
# Verify the harness end-to-end before writing any agent code.
#
# The oracle agent runs each task's reference solution, so a green run means
# "Docker + Harbor + the dataset all work here" and a red one is an
# environment problem, never an agent problem. Run this first, and any time
# the toolchain changes.
set -euo pipefail

# Trial output is thousands of small files. On the cluster that must land on
# the local NVMe, never on the shared cpfs mount the code lives on — cpfs
# would make the writes, not the model, the bottleneck.
if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

DATASET="${DATASET:-terminal-bench/terminal-bench@latest}"
N_TASKS="${N_TASKS:-3}"
N_CONCURRENT="${N_CONCURRENT:-2}"

echo "dataset : ${DATASET}"
echo "jobs    : ${JOBS_DIR}"
echo "tasks   : ${N_TASKS} (oracle / reference solutions)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent oracle \
  --n-tasks "${N_TASKS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env docker \
  --yes
