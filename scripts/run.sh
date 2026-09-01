#!/usr/bin/env bash
# Run the benchmark. A thin wrapper over `crux bench`, which is the one
# implementation.
#
# This script used to build the harbor command line itself, in parallel with
# cli.py doing the same thing. They drifted, and the same bug had to be found
# twice: both hardcoded a `terminal-bench/` prefix on the GPU-task exclusion, so
# on any other dataset the exclusion silently matched nothing while still
# printing what it had "skipped".
#
# Environment variables are kept because the ablation scripts and the remote
# runners set them; each maps onto a flag.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=()
[ -n "${DATASET:-}" ]      && ARGS+=(--dataset "${DATASET}")
[ -n "${AGENT:-}" ]        && ARGS+=(--agent "${AGENT}")
[ -n "${MODEL:-}" ]        && ARGS+=(--model "${MODEL}")
[ -n "${VARIANT:-}" ]      && ARGS+=(--variant "${VARIANT}")
[ -n "${N_TASKS:-}" ]      && ARGS+=(--tasks "${N_TASKS}")
[ -n "${N_ATTEMPTS:-}" ]   && ARGS+=(--attempts "${N_ATTEMPTS}")
[ -n "${N_CONCURRENT:-}" ] && ARGS+=(--concurrent "${N_CONCURRENT}")
[ -n "${JOBS_DIR:-}" ]     && ARGS+=(--jobs-dir "${JOBS_DIR}")
[ -n "${ENV_FILE:-}" ]     && ARGS+=(--env-file "${ENV_FILE}")
[ -n "${UPLOAD:-}" ]       && ARGS+=(--upload)
# Space separated, e.g. AK_EXTRA="model_api=openai-completions thinking=1"
for kv in ${AK_EXTRA:-}; do ARGS+=(--ak "${kv}"); done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages 2>/dev/null | head -1)
export PYTHONPATH="$(pwd)/src${SP:+:${SP}}"

exec python3 -m crux.cli bench "${ARGS[@]}" "$@"
