#!/usr/bin/env bash
# Verify the harness end-to-end before writing any agent code.
#
# The oracle agent runs each task's reference solution, so a green run means
# "Docker + Harbor + the dataset all work here" and a red one is an
# environment problem, never an agent problem. Run this first, and any time
# the toolchain changes.
#
# Deliberately the one script that still builds its own harbor command line
# rather than going through `crux bench`. A check that tells you whether the box
# is sound must not stop working when crux does.
set -euo pipefail

# Trial output is thousands of small files. On the cluster that must land on
# the local NVMe, never on the shared cpfs mount the code lives on — cpfs
# would make the writes, not the model, the bottleneck.
if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

# Pinned rather than @latest: this check exists to say whether the box is sound,
# and @latest silently became 3.0 and then 4.0 while the surrounding comments
# still described an older set. A moving target cannot tell you whether
# something changed here or upstream.
DATASET="${DATASET:-terminal-bench/terminal-bench@4.0.0}"
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
