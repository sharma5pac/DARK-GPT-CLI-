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
DEFAULT_TIMEOUT = 120


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
    def _norm(a: Any) -> Optional[Dict[str, Any]]:
        if isinstance(a, str):
            if a.startswith(("http://", "https://", "file://", "data:")):
                return {"action": "navigate", "url": a}
            return {"action": "extract", "selector": a}
        if not isinstance(a, dict):
            return None
        d = dict(a)
        if d.get("action"):
            return d
        # Tolerate shorthand shapes the model often emits.
        for k in ("navigate", "goto", "open", "url"):
            if k in d and isinstance(d[k], str):
                return {"action": "navigate", "url": d[k], **{x: y for x, y in d.items() if x != k}}
        if "wait" in d:
            w = d["wait"]
            if isinstance(w, dict):
                return {"action": "wait", **w}
            if isinstance(w, (int, float)):
                return {"action": "wait", "ms": int(w)}
            if isinstance(w, str):
                return {"action": "wait", "selector": w}
        for k in ("click", "fill", "extract", "screenshot", "scroll", "js", "crawl", "rank", "search"):
            if k in d:
                v = d[k]
                if isinstance(v, dict):
                    return {"action": k, **v}
                if k == "click":
                    return {"action": "click", "selector": str(v) if not str(v).startswith("//") else "", "text": "", "xpath": str(v) if str(v).startswith("//") else ""}
                if k == "fill":
                    return {"action": "fill", **(v if isinstance(v, dict) else {"selector": "", "value": str(v)})}
                if k == "extract":
                    return {"action": "extract", "selector": str(v)}
                if k == "screenshot":
                    return {"action": "screenshot", "path": str(v)}
                if k == "scroll":
                    return {"action": "scroll", "direction": str(v)}
                if k == "js":
                    return {"action": "js", "script": str(v)}
                if k in ("crawl", "rank", "search"):
                    return {"action": k, "query": str(v) if not isinstance(v, dict) else "", **(v if isinstance(v, dict) else {})}
        if "ms" in d or "seconds" in d:
            return {"action": "wait", **d}
        if "selector" in d and "value" in d:
            return {"action": "fill", **d}
        if "selector" in d or "mode" in d:
            return {"action": "extract", **d}
        return d

    for item in actions or []:
        n = _norm(item)
        if n:
            steps.append(n)
    if not steps:
        return {"ok": False, "error": "no browser actions supplied"}

    # Clamp up: agents often pass small timeouts; slow/JS-heavy sites need room.
    timeout = max(45, min(int(timeout or DEFAULT_TIMEOUT), 180))
    state = _load_session(str(session)[:40] or "default")
    job = {
        "actions": steps,
        "engine": str(engine or "auto"),
        "headless": bool(headless),
        "nav_timeout": timeout,
        "cookies": state.get("cookies") or [],
        "storage": state.get("storage") or None,
    }
    payload = json.dumps(job, ensure_ascii=False)
    py = _system_python()
    try:
        proc = subprocess.run(
            [py, str(WORKER), payload],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=str(WORKER.parent.parent),
        )
    except subprocess.TimeoutExpired as exc:
        # One graceful fallback: simple HTTP fetch snapshot so agent is not blind.
        fallback = {}
        try:
            import urllib.request
            nav = next((s.get("url") for s in steps if isinstance(s, dict) and s.get("action") == "navigate" and str(s.get("url","")).startswith(("http://","https://"))), None)
            if nav:
                req = urllib.request.Request(nav, headers={"User-Agent": "Mozilla/5.0 DarkGPT/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    body = r.read(250000).decode("utf-8", errors="replace")
                fallback = {"fallback": "http_fetch", "url": nav, "status": getattr(r, "status", 200), "body_preview": body[:12000]}
        except Exception:
            fallback = {}
        return {
            "ok": False,
            "error": f"browser timed out after {timeout}s",
            "engine": str(engine or "auto"),
            "hint": "Browser timed out. Pivot: use fetch_web_content/curl, try engine='selenium', increase timeout, or use browser action crawl/rank on a simpler page. A lightweight HTTP fallback may be included below.",
            **fallback,
            "partial": str(exc)[:200],
        }
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
