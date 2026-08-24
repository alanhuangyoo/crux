#!/usr/bin/env bash
# Phase 1 — establish the starting line.
#
# Runs an existing Harbor agent against DeepSeek so later scaffold work has
# something honest to be measured against. Deliberately a small subset: the
# point is a fast comparable number, not a leaderboard-grade run.
set -euo pipefail

# Trial output is thousands of small files. On the cluster that must land on
# the local NVMe, never on the shared cpfs mount the code lives on — cpfs
# would make the writes, not the model, the bottleneck.
if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

DATASET="${DATASET:-terminal-bench/terminal-bench@latest}"
AGENT="${AGENT:-mini-swe-agent}"
MODEL="${MODEL:-deepseek/deepseek-chat}"
N_TASKS="${N_TASKS:-10}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-4}"

echo "dataset : ${DATASET}"
echo "agent   : ${AGENT}"
echo "model   : ${MODEL}"
echo "jobs    : ${JOBS_DIR}"
echo "tasks   : ${N_TASKS} x ${N_ATTEMPTS} attempt(s)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --n-tasks "${N_TASKS}" \
  --n-attempts "${N_ATTEMPTS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --jobs-dir "${JOBS_DIR}" \
  --env docker \
  --yes
