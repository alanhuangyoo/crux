#!/usr/bin/env bash
# Push the working tree to the dev box.
#
# Evaluation runs on dev, not on this Mac: the tasks ship x86-64 binaries and
# this laptop is arm64, so local scores are meaningless (see docs/ENVIRONMENT.md).
# Write code here, run it there.
set -euo pipefail

REMOTE="${REMOTE:-dev}"
REMOTE_DIR="${REMOTE_DIR:-~/crux/}"

rsync -az --delete \
  --exclude 'jobs/' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  -e ssh ./ "${REMOTE}:${REMOTE_DIR}"

echo "synced -> ${REMOTE}:${REMOTE_DIR}"
