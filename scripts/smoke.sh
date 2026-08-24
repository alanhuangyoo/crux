#!/usr/bin/env bash
# Verify the harness end-to-end before writing any agent code.
#
# The oracle agent runs each task's reference solution, so a green run means
# "Docker + Harbor + the dataset all work here" and a red one is an
# environment problem, never an agent problem. Run this first, and any time
# the toolchain changes.
set -euo pipefail

DATASET="${DATASET:-terminal-bench/terminal-bench@latest}"
N_TASKS="${N_TASKS:-3}"
N_CONCURRENT="${N_CONCURRENT:-2}"

echo "dataset : ${DATASET}"
echo "tasks   : ${N_TASKS} (oracle / reference solutions)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent oracle \
  --n-tasks "${N_TASKS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --env docker \
  --yes
