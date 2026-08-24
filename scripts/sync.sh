#!/usr/bin/env bash
# Push the working tree to the eval box.
#
# Evaluation runs on h20-43, not on this Mac: the tasks ship x86-64 binaries
# and this laptop is arm64, so local scores are meaningless (docs/ENVIRONMENT.md).
# Write code here, run it there.
set -euo pipefail

REMOTE="${REMOTE:-h20-43}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/cpfs/users/xiaohuang/projects/crux/}"

# .env holds the API key and lives only on the remote. Without this exclude,
# --delete removes it on every sync because it has no local counterpart.
rsync -az --delete \
  --exclude '.env' \
  --exclude 'jobs/' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  -e ssh ./ "${REMOTE}:${REMOTE_DIR}"

echo "synced -> ${REMOTE}:${REMOTE_DIR}"
