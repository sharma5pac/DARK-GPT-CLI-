"""Built-in browser tool for DarkGPT.

The model calls this with high-level actions; no manual script writing. A
small worker runs in system Python (where Selenium/Playwright are installed)
and returns structured results plus session cookies so state survives across
calls. Playwright is tried first, Selenium is the automatic fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SESSION_DIR = Path.home() / ".cache" / "darkgpt" / "browser"
WORKER = Path(__file__).with_name("browser_worker.py")
DEFAULT_TIMEOUT = 90


def _system_python() -> str:
    """Prefer the interpreter that actually has Selenium/Playwright."""
    candidates = ["/usr/bin/python3", shutil.which("python3"), sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                probe = subprocess.run(
                    [candidate, "-c", "import selenium"],
                    capture_output=True,
                    timeout=6,
                )
                if probe.returncode == 0:
                    return str(candidate)
            except Exception:
                continue
    return sys.executable


def _load_session(session: str) -> Dict[str, Any]:
    try:
        path = SESSION_DIR / f"{session}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_session(session: str, data: Dict[str, Any]) -> None:
    try:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSION_DIR / f"{session}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def run(
    actions: Any,
    engine: str = "auto",
    session: str = "default",
    timeout: int = DEFAULT_TIMEOUT,
    headless: bool = True,
) -> Dict[str, Any]:
    """Execute browser actions and return a compact structured result."""
    if isinstance(actions, dict):
        actions = [actions]
    steps: List[Dict[str, Any]] = []
    for item in actions or []:
        if isinstance(item, str):
            steps.append({"action": "navigate", "url": item} if item.startswith(("http://", "https://")) else {"action": "extract", "selector": item})
        elif isinstance(item, dict):
            steps.append(item)
    if not steps:
        return {"ok": False, "error": "no browser actions supplied"}

    state = _load_session(str(session)[:40] or "default")
    job = {
        "actions": steps,
        "engine": str(engine or "auto"),
        "headless": bool(headless),
        "nav_timeout": max(5, min(int(timeout or DEFAULT_TIMEOUT), 180)),
        "cookies": state.get("cookies") or [],
        "storage": state.get("storage") or None,
    }
    payload = json.dumps(job, ensure_ascii=False)
    try:
        proc = subprocess.run(
            [_system_python(), str(WORKER), payload],
            capture_output=True,
            text=True,
            timeout=max(20, int(timeout or DEFAULT_TIMEOUT) + 20),
            cwd=str(WORKER.parent.parent),
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"browser timed out after {timeout}s", "partial": str(exc)[:200]}
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout or "browser worker failed")[-2000:],
        }
    try:
        result = json.loads(proc.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"invalid browser result: {exc}", "raw": proc.stdout[-1000:]}

    if isinstance(result, dict):
        state = {}
        if result.get("cookies"):
            state["cookies"] = result["cookies"]
        if result.get("storage"):
            state["storage"] = result["storage"]
        if state:
            _save_session(str(session)[:40] or "default", state)
    return result
