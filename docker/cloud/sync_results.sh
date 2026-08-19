#!/usr/bin/env bash
# Pulls finished checkpoints (and anything exported to .onnx) back from the
# remote cloud box to this machine. Run LOCALLY. Never uploads anything --
# read-only pull.
#
# Usage:
#   ./sync_results.sh user@remote-host [remote_robot_lab_path]
#
# Defaults assume the container was run with
#   -v $(pwd)/cloud_logs:/workspace/robot_lab/logs
# on the remote host, i.e. results live under <remote_robot_lab_path>/logs
# on the HOST filesystem there (not just inside the container).

set -euo pipefail

REMOTE="${1:?usage: sync_results.sh user@remote-host [remote_robot_lab_path]}"
REMOTE_PATH="${2:-~/robot_lab}"
LOCAL_DEST="$(dirname "$0")/../../cloud_results"

mkdir -p "$LOCAL_DEST"

echo "Pulling checkpoints (model_*.pt) and exported onnx from ${REMOTE}:${REMOTE_PATH}/logs ..."
rsync -avz --progress \
  --include='*/' \
  --include='model_*.pt' \
  --include='*.onnx' \
  --include='*.yaml' \
  --include='events.out.tfevents*' \
  --exclude='*' \
  "${REMOTE}:${REMOTE_PATH}/logs/" "${LOCAL_DEST}/"

echo "Done. Results in ${LOCAL_DEST}"
