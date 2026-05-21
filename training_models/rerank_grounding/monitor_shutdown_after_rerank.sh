#!/usr/bin/env bash
set -euo pipefail
PID="$1"
SUMMARY="$2"
LOG="$3"
echo "Monitor started: $(date)" | tee -a "$LOG"
echo "Watching PID=$PID" | tee -a "$LOG"
echo "Expect summary=$SUMMARY" | tee -a "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done
echo "Rerank PID exited: $(date)" | tee -a "$LOG"
if [ -s "$SUMMARY" ]; then
  echo "Summary exists. Auto shutdown now." | tee -a "$LOG"
  sync
  shutdown -h now
else
  echo "Summary missing; leaving server running for debugging." | tee -a "$LOG"
fi
