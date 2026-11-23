#!/usr/bin/env bash
set -euo pipefail
if systemctl --user is-active --quiet darkgpt.service 2>/dev/null; then
  systemctl --user stop darkgpt.service
  echo "Stopped darkgpt.service"
else
  # fallback: kill uvicorn on 8501
  pid=$(ss -ltnp 2>/dev/null | sed -n 's/.*:8501 .*pid=\([0-9]*\).*/\1/p' | head -1)
  if [ -n "${pid:-}" ]; then kill "$pid" && echo "Stopped pid $pid"; else echo "Not running"; fi
fi
