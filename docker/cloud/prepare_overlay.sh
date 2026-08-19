#!/usr/bin/env bash
# Regenerates docker/cloud/b2_overlay/ -- the ONLY files that carry our B2
# training work into the cloud build. Run this LOCALLY (never on the remote
# machine) whenever the B2 task configs change, before rebuilding the cloud
# image.
#
# What this does: extracts exactly the files that differ between our fork
# (HEAD) and the public upstream (origin/main) via `git show`, i.e. content
# only -- no .git history, no commit messages, no other files. This is
# deliberately NOT `git clone`/`cp -r` of the whole repo: those would carry
# every .md file, every commit ever made (including anything since removed
# from HEAD but still in history), and generally far more than a cloud GPU
# box building our task configs needs to see.
#
# Safety check built in: refuses to write anything if the diff against
# upstream ever picks up a .md file, a CLAUDE.md, or anything under a
# memory/ path -- those must NEVER end up in this directory. If this ever
# fires, STOP and figure out why the diff grew before touching anything
# further (it means either upstream moved, or something unexpected got
# committed to our fork's tracked b2 files).

set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root (robot_lab)

OVERLAY_DIR="docker/cloud/b2_overlay"
rm -rf "$OVERLAY_DIR"
mkdir -p "$OVERLAY_DIR"

FILES=$(git diff --name-only origin/main...HEAD)

for f in $FILES; do
  case "$f" in
    *.md|*CLAUDE*|*memory/*|*research/*|*README*)
      echo "REFUSING: '$f' looks like docs/notes, not a training config. Aborting." >&2
      rm -rf "$OVERLAY_DIR"
      exit 1
      ;;
  esac
  mkdir -p "$OVERLAY_DIR/$(dirname "$f")"
  git show HEAD:"$f" > "$OVERLAY_DIR/$f"
done

echo "Overlay staged in $OVERLAY_DIR ($(echo "$FILES" | wc -l) files):"
echo "$FILES"
echo
echo "NOTE: these files' own comments quote live bench feedback (Russian,"
echo "e.g. owner verdicts explaining why a reward weight is what it is) --"
echo "that's the actual calibration rationale, deliberately kept (it's part"
echo "of the config, not a diary). If that's unwanted on the cloud box too,"
echo "say so explicitly before building -- this script does not strip it."
