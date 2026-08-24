#!/usr/bin/env bash
# Phase 1 — establish the starting line.
#
# Runs an existing Harbor agent against DeepSeek so later scaffold work has
# something honest to be measured against. Deliberately a small subset: the
# point is a fast comparable number, not a leaderboard-grade run.
set -euo pipefail

DATASET="${DATASET:-terminal-bench/terminal-bench@latest}"
AGENT="${AGENT:-mini-swe-agent}"
MODEL="${MODEL:-deepseek/deepseek-chat}"
N_TASKS="${N_TASKS:-10}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
N_CONCURRENT="${N_CONCURRENT:-4}"

echo "dataset : ${DATASET}"
echo "agent   : ${AGENT}"
echo "model   : ${MODEL}"
echo "tasks   : ${N_TASKS} x ${N_ATTEMPTS} attempt(s)"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --n-tasks "${N_TASKS}" \
  --n-attempts "${N_ATTEMPTS}" \
  --n-concurrent "${N_CONCURRENT}" \
  --env docker \
  --yes
