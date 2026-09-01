#!/usr/bin/env bash
# The development slice. A wrapper over `crux quick`, which is the one
# implementation.
#
# This was one of five places that built the harbor command line by hand, and
# it had drifted the same way the others did: it named crux.agent:CruxAgent, the
# base every measurement rejected, and defaulted to a hosted model no run here
# has used since the engine moved onto four H20s.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS=()
[ -n "${AGENT:-}" ]        && ARGS+=(--agent "${AGENT}")
[ -n "${MODEL:-}" ]        && ARGS+=(--model "${MODEL}")
[ -n "${VARIANT:-}" ]      && ARGS+=(--variant "${VARIANT}")
[ -n "${N_CONCURRENT:-}" ] && ARGS+=(--concurrent "${N_CONCURRENT}")
[ -n "${JOBS_DIR:-}" ]     && ARGS+=(--jobs-dir "${JOBS_DIR}")
for kv in ${AK_EXTRA:-}; do ARGS+=(--ak "${kv}"); done

SP=$(ls -d "$HOME"/.local/share/uv/tools/harbor/lib/python3.*/site-packages 2>/dev/null | head -1)
export PYTHONPATH="$(pwd)/src${SP:+:${SP}}"
exec python3 -m crux.cli quick "${ARGS[@]}" "$@"
