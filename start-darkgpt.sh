#!/usr/bin/env bash
set -euo pipefail

SERVICE="darkgpt.service"
URL="http://127.0.0.1:8501"
DIR="$HOME/Desktop/darkgpt/DarkGPT"
VENV="$DIR/.venv/bin/uvicorn"

echo "== DarkGPT =="
if command -v systemctl >/dev/null 2>&1 && systemctl --user list-unit-files "$SERVICE" >/dev/null 2>&1; then
  if systemctl --user is-active --quiet "$SERVICE"; then
    echo "Already running (systemd user service)."
  else
    echo "Starting systemd user service..."
    systemctl --user start "$SERVICE"
    sleep 2
  fi
  systemctl --user --no-pager --full status "$SERVICE" | sed -n '1,12p'
else
  if ss -ltn 2>/dev/null | grep -q ':8501 '; then
    echo "Port 8501 already listening."
  else
    echo "Starting uvicorn directly..."
    cd "$DIR"
    nohup "$VENV" server.api:app --host 127.0.0.1 --port 8501 --log-level warning > /tmp/darkgpt.log 2>&1 &
    sleep 2
  fi
fi

echo
echo "URL: $URL"
if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -w "%{http_code}" "$URL/api/health" || true)
  echo "Health: $code"
fi
