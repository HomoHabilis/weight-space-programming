#!/usr/bin/env bash
# Stops any single `claude -p` call from the agent harness that has run longer than LIMIT seconds (a hung connection);
# the harness then retries the same turn. Usage: scripts/watchdog_claude_calls.sh [limit_seconds=1200]
LIMIT=${1:-1200}
while true; do
  ps -eo pid=,etimes=,args= | awk -v lim="$LIMIT" '$3 ~ /claude$/ && $4 == "-p" && $2 > lim {print $1, $2}' |
  while read -r pid secs; do
    if [ -d "/proc/$pid" ] && [[ "$(readlink /proc/$pid/cwd)" == /tmp/wsp-claude-* ]]; then
      echo "$(date '+%F %T') watchdog: stopping claude call $pid after ${secs}s"; kill -TERM "$pid"
    fi
  done
  sleep 30
done
