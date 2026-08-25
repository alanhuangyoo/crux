#!/usr/bin/env bash
# Measure a built-in Harbor agent on the same tasks and model as Crux.
#
# Crux is only worth having if it beats the scaffolds that already exist.
# Terminus 2 and mini-SWE-agent both sit on the public leaderboard (80.4% and
# 76.2%), so they are the honest reference point -- and if one of them is far
# ahead, the right move is to build on it rather than keep tuning our own loop.
set -euo pipefail

if [ -z "${JOBS_DIR:-}" ]; then
  if [ -d /scratch ]; then JOBS_DIR=/scratch/crux-jobs; else JOBS_DIR=jobs; fi
fi

AGENT="${AGENT:-terminus-2}"
DATASET="${DATASET:-terminal-bench/terminal-bench-2-1}"
MODEL="${MODEL:-deepseek/deepseek-v4-flash}"
N_TASKS="${N_TASKS:-12}"
N_CONCURRENT="${N_CONCURRENT:-12}"

# Same exclusions as run.sh: these need a GPU, and the validation error aborts
# the whole job rather than just those trials.
GPU_TASKS="${GPU_TASKS:-exam-pdf-eval fp8-rmsnorm-gemm jax-speedrun-gpu math-eval-grader}"
EXCLUDE=""
for t in ${GPU_TASKS}; do EXCLUDE="${EXCLUDE} --exclude-task-name terminal-bench/${t}"; done

echo "agent   : ${AGENT}"
echo "model   : ${MODEL}"
echo "tasks   : ${N_TASKS}   concurrent: ${N_CONCURRENT}"
echo

harbor run \
  --dataset "${DATASET}" \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --n-tasks "${N_TASKS}" \
  --n-concurrent "${N_CONCURRENT}" \
  ${EXCLUDE} \
  --jobs-dir "${JOBS_DIR}" \
  --env-file .env \
  --env docker \
  --yes
