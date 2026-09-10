#!/usr/bin/env bash
# Measure a built-in Harbor agent on the same tasks and model as Crux.
#
# Crux is only worth having if it beats the scaffolds that already exist, so
# terminus-2, claude-code and pi are the reference points -- and if one is far
# ahead, the right move is to build on it rather than keep tuning our own loop.
#
# A wrapper over `crux bench --agent`, which grew that flag precisely so a
# comparison arm runs through the same code path with the same dataset, model,
# concurrency and attempts. Assembling it separately is how this script came to
# default to a model no run has used in weeks.
set -euo pipefail
cd "$(dirname "$0")/.."

AGENT="${AGENT:-terminus-2}"
ARGS=(--agent "${AGENT}")
[ -n "${DATASET:-}" ]      && ARGS+=(--dataset "${DATASET}")
[ -n "${MODEL:-}" ]        && ARGS+=(--model "${MODEL}")
[ -n "${N_TASKS:-}" ]      && ARGS+=(--tasks "${N_TASKS}")
[ -n "${N_ATTEMPTS:-}" ]   && ARGS+=(--attempts "${N_ATTEMPTS}")
[ -n "${N_CONCURRENT:-}" ] && ARGS+=(--concurrent "${N_CONCURRENT}")
[ -n "${JOBS_DIR:-}" ]     && ARGS+=(--jobs-dir "${JOBS_DIR}")
[ -n "${AGENT_TIMEOUT_MULTIPLIER:-}" ] && ARGS+=(--agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER}")
for kv in ${AK_EXTRA:-}; do ARGS+=(--ak "${kv}"); done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages 2>/dev/null | head -1)
export PYTHONPATH="$(pwd)/src${SP:+:${SP}}"
exec python3 -m crux.cli bench "${ARGS[@]}" "$@"
