#!/usr/bin/env bash
# Regenerate the RQ4 summary every 15 minutes while the run is in flight.
#
# The point is that a fresh summary always exists, whatever hour someone looks.
# The report is written to be safe on a partial run — it counts complete epochs
# and withholds the primary result until at least one epoch is complete for all
# three arms — so running it early costs nothing and hides nothing.
set -u

RUN_ID="${RUN_ID:-rq4-2026-08-09}"
EXPECTED="${EXPECTED:-810}"
REPO=/home/mosyamac/b2e-synt-data
TURNS="$REPO/var/rq4/$RUN_ID/turns.jsonl"
LOG="$REPO/var/rq4/$RUN_ID/report-loop.log"

cd "$REPO" || exit 1

while :; do
  rows=0
  [ -f "$TURNS" ] && rows=$(wc -l < "$TURNS")
  PYTHONPATH=. .venv/bin/python scripts/rq4_report.py --run-id "$RUN_ID" \
    >> "$LOG" 2>&1
  echo "$(date -u +%FT%TZ) regenerated at $rows/$EXPECTED rows" >> "$LOG"
  [ "$rows" -ge "$EXPECTED" ] && break
  sleep 900
done

echo "$(date -u +%FT%TZ) report loop exit" >> "$LOG"
