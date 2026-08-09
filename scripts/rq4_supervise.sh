#!/usr/bin/env bash
# Keep the RQ4 driver alive until the run is complete.
#
# The driver is resumable — turns.jsonl is also its checkpoint — so the recovery
# action for any crash is simply to start it again with the same --run-id. What
# this adds is nobody having to be awake to do that. It stops when the expected
# row count is reached, or after too many consecutive restarts, because a driver
# that dies instantly forever is a bug to look at rather than a thing to retry.
set -u

RUN_ID="${RUN_ID:-rq4-2026-08-09}"
EXPECTED="${EXPECTED:-810}"
MAX_RESTARTS="${MAX_RESTARTS:-40}"
REPO=/home/mosyamac/b2e-synt-data
TURNS="$REPO/var/rq4/$RUN_ID/turns.jsonl"
LOG="$REPO/var/rq4/$RUN_ID/supervisor.log"

say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

restarts=0
say "supervisor start run_id=$RUN_ID expected=$EXPECTED"

while :; do
  rows=0
  [ -f "$TURNS" ] && rows=$(wc -l < "$TURNS")
  if [ "$rows" -ge "$EXPECTED" ]; then
    say "complete: $rows/$EXPECTED rows"
    break
  fi

  if docker ps --filter name=rq4-run --format '{{.Names}}' | grep -q rq4-run; then
    sleep 60
    continue
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "giving up after $restarts restarts at $rows/$EXPECTED rows"
    break
  fi

  restarts=$((restarts + 1))
  say "driver not running at $rows/$EXPECTED rows — restart #$restarts"
  docker rm -f rq4-run >/dev/null 2>&1
  docker run -d --name rq4-run --network b2e-sim_internal \
    -v "$REPO":/app -v b2e-sim_registry_data:/app/registry \
    --env-file "$REPO/deploy/.env" -w /app b2e-sim/agent:local \
    python -u scripts/run_rq4.py --seed 20260809 --hr-employee-id 9877478 \
      --replications 1 --epochs 9 --concurrency 8 --turn-timeout-seconds 2400 \
      --skip-spans --run-id "$RUN_ID" >/dev/null 2>&1
  docker network connect b2e-sim_edge rq4-run >/dev/null 2>&1
  sleep 90
done

say "supervisor exit"
